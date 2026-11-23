import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.simulation_config import SimulationConfig
import scripts.generate_mesh as gm
from scripts.post_process import (
    SAMPLING_END_PCT,
    SAMPLING_START_PCT,
    extract_cfd_gradient_and_holdup,
    extract_yplus,
    get_mesh_cells,
)
from scripts.setup_0 import setup_zero_files
from scripts.setup_constant import setup_constant_files
from scripts.setup_system import setup_system_files


# Razão r ≈ 1.414 (> 1.3 recomendado pela norma ASME GCI)
GRID_FACTORS = {
    "Grossa": 0.707,
    "Média": 1.000,
    "Fina": 1.414,
}
WALL_EXPANSION_RATIO = 8.0
SHORT_END_TIME = 1.5
OUTPUT_CSV = ROOT_DIR / "result_GI.csv"


def get_of_env() -> str:
    possible_paths = [
        "/usr/lib/openfoam/openfoam2606/etc/bashrc",
        "/opt/openfoam2606/etc/bashrc",
        os.path.expanduser("~/OpenFOAM/openfoam2606/etc/bashrc"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return f"source {path} && "
    return "source /usr/lib/openfoam/openfoam2606/etc/bashrc && "


OF_ENV = get_of_env()


def run_cmd(cmd: str, cwd: Path):
    """Executa comandos bash no ambiente do OpenFOAM."""
    env = os.environ.copy()
    subprocess.run(
        f"{OF_ENV} {cmd}",
        shell=True,
        executable="/bin/bash",
        cwd=cwd,
        check=True,
        env=env,
    )


def extract_velocity_metrics(case_dir: Path, num_cores: int) -> tuple[float, float]:
    """Extrai U_avg e U_max (m/s) do perfil radial (profile_mid) no último tempo."""
    sets_dir = case_dir / "postProcessing" / "sets"
    if sets_dir.exists():
        shutil.rmtree(sets_dir, ignore_errors=True)

    has_processors = any(case_dir.glob("processor*"))
    exec_cmd = (
        f"mpirun -np {num_cores} postProcess -func sampleDict -latestTime -parallel"
        if has_processors
        else "postProcess -func sampleDict -latestTime"
    )
    subprocess.run(
        f"{OF_ENV} {exec_cmd}",
        shell=True,
        executable="/bin/bash",
        cwd=case_dir,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    profile_files = sorted(
        [f for f in sets_dir.rglob("*profile_mid*U*") if f.is_file()]
    ) if sets_dir.exists() else []
    if not profile_files:
        return 0.0, 0.0

    try:
        data = np.loadtxt(profile_files[-1], comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Formato raw de vetor: [coord, ux, uy, uz]
        if data.shape[1] >= 4:
            u_mag = np.linalg.norm(data[:, 1:4], axis=1)
            return float(np.mean(u_mag)), float(np.max(u_mag))
    except Exception:
        pass
    return 0.0, 0.0


def compute_asme_gci(phi_coarse: float, phi_med: float, phi_fine: float,
                     n_coarse: int, n_med: int, n_fine: int) -> dict:
    """
    Calcula o Grid Convergence Index (GCI) segundo a metodologia ASME (Celik et al., 2008).
    Índices padrão ASME: 1 = Fina, 2 = Média, 3 = Grossa.
    """
    r21 = (n_fine / n_med) ** (1.0 / 3.0) if n_med > 0 else 1.414
    r32 = (n_med / n_coarse) ** (1.0 / 3.0) if n_coarse > 0 else 1.414

    eps32 = phi_coarse - phi_med
    eps21 = phi_med - phi_fine

    ea_21 = abs((phi_fine - phi_med) / phi_fine) * 100.0 if phi_fine != 0 else 0.0
    ea_32 = abs((phi_med - phi_coarse) / phi_med) * 100.0 if phi_med != 0 else 0.0

    # Caso degenerado (diferença nula)
    if abs(eps21) < 1e-12 or abs(eps32) < 1e-12:
        return {
            "r21": r21,
            "r32": r32,
            "p_order": 0.0,
            "ea_21_pct": ea_21,
            "ea_32_pct": ea_32,
            "gci_fine_pct": 0.0,
            "gci_med_pct": 0.0,
            "asymptotic_ratio": 1.0,
            "phi_ext_21": phi_fine,
            "convergence_type": "Convergido (diferença nula)",
        }

    s = 1.0 if (eps32 / eps21) > 0 else -1.0
    conv_type = "Monotônica" if s > 0 else "Oscilatória"

    # Iteração de ponto fixo para ordem aparente p (Celik et al., 2008)
    p = abs(math.log(abs(eps32 / eps21))) / math.log(r21)
    for _ in range(50):
        denom = (r32 ** p) - s
        if abs(denom) < 1e-12 or ((r21 ** p) - s) <= 0:
            break
        q = math.log(abs(((r21 ** p) - s) / denom))
        p_new = abs(math.log(abs(eps32 / eps21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-6:
            p = p_new
            break
        p = p_new

    # Limita p para evitar explosão numérica em dados transientes ruidosos
    p = max(0.5, min(p, 4.0))

    fs = 1.25  # Fator de segurança ASME para 3+ malhas
    gci_fine = (fs * ea_21) / max((r21 ** p) - 1.0, 1e-6)
    gci_med = (fs * ea_32) / max((r32 ** p) - 1.0, 1e-6)
    phi_ext_21 = ((r21 ** p) * phi_fine - phi_med) / max((r21 ** p) - 1.0, 1e-6)
    asymp_ratio = gci_med / max((r21 ** p) * gci_fine, 1e-9)

    return {
        "r21": r21,
        "r32": r32,
        "p_order": p,
        "ea_21_pct": ea_21,
        "ea_32_pct": ea_32,
        "gci_fine_pct": gci_fine,
        "gci_med_pct": gci_med,
        "asymptotic_ratio": asymp_ratio,
        "phi_ext_21": phi_ext_21,
        "convergence_type": conv_type,
    }


def update_simulation_config_factor(recommended_factor: float):
    """Atualiza 'mesh_factor' em config/simulation_config.py com o fator ideal."""
    config_path = ROOT_DIR / "config" / "simulation_config.py"
    if not config_path.exists():
        print(f"[-] Arquivo não encontrado: {config_path}")
        return

    content = config_path.read_text(encoding="utf-8")
    updated_content = re.sub(
        r"(mesh_factor\s*:\s*float\s*=\s*|mesh_factor\s*=\s*)([\d\.]+)",
        rf"\g<1>{recommended_factor:.3f}",
        content,
    )
    config_path.write_text(updated_content, encoding="utf-8")
    print(f"[+] 'mesh_factor = {recommended_factor:.3f}' atualizado em config/simulation_config.py")


def run_grid_study():
    base_cfg = SimulationConfig()
    z_start = base_cfg.length * (SAMPLING_START_PCT / 100.0)
    z_end = base_cfg.length * (SAMPLING_END_PCT / 100.0)
    num_cores = base_cfg.num_processors

    results = []
    total_start_time = time.time()

    print("=" * 90)
    print(" ESTUDO DE INDEPENDÊNCIA DE MALHA — NORMA ASME GCI (Celik et al., 2008)")
    print("=" * 90)

    for name, factor in GRID_FACTORS.items():
        cfg = base_cfg.model_copy(update={"mesh_factor": factor, "end_time": SHORT_END_TIME})
        n_core = max(8, int(10 * factor))
        n_radial = max(16, int(20 * factor))
        nz = max(10, int(100 * factor))

        case_dir = ROOT_DIR / "runs" / f"grid_{name.lower().replace('é', 'e')}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[+] Malha {name} (factor={factor:.3f} | n_core={n_core}, n_radial={n_radial}, nz={nz})...")
        grid_start_time = time.time()

        # 1. Gera todos os dicionários diretamente para a malha atual
        setup_system_files(cfg, case_dir)
        setup_constant_files(cfg, case_dir)
        setup_zero_files(cfg, case_dir)
        gm.generate_blockmesh_dict(cfg, case_dir / "system" / "blockMeshDict", wall_expansion_ratio=WALL_EXPANSION_RATIO)

        # 2. Malha, decomposição, simulação e reconstrução
        run_cmd("blockMesh > log.blockMesh 2>&1", case_dir)
        run_cmd("decomposePar -force > log.decomposePar 2>&1", case_dir)
        run_cmd(f"mpirun -np {num_cores} interFoam -parallel > log.interFoam 2>&1", case_dir)
        run_cmd("reconstructPar > log.reconstructPar 2>&1", case_dir)

        grid_elapsed = time.time() - grid_start_time
        n_cells = get_mesh_cells(case_dir)

        # 3. Extração de métricas físicas e de parede
        cfd_metrics = extract_cfd_gradient_and_holdup(case_dir, z_start, z_end, num_cores)
        yplus = extract_yplus(case_dir, num_cores)
        u_avg, u_max = extract_velocity_metrics(case_dir, num_cores)

        results.append({
            "Malha": name,
            "Mesh_Factor": factor,
            "n_core": n_core,
            "n_radial": n_radial,
            "nz": nz,
            "Mesh_Cells": n_cells,
            "Grad_P_CFD_Pa_m": round(cfd_metrics["grad_p_CFD_Pa_m"], 4),
            "Holdup_CFD": round(cfd_metrics["holdup_CFD_avg"], 4),
            "U_Avg_m_s": round(u_avg, 4),
            "U_Max_m_s": round(u_max, 4),
            "yPlus_Min": round(yplus["min"], 2),
            "yPlus_Max": round(yplus["max"], 2),
            "yPlus_Avg": round(yplus["avg"], 2),
            "yPlus_Status": yplus["status"],
            "Execution_Time_s": round(grid_elapsed, 2),
        })

    # 4. Cálculo do ASME GCI (Grossa=0, Média=1, Fina=2)
    gci_dp = compute_asme_gci(
        results[0]["Grad_P_CFD_Pa_m"],
        results[1]["Grad_P_CFD_Pa_m"],
        results[2]["Grad_P_CFD_Pa_m"],
        results[0]["Mesh_Cells"],
        results[1]["Mesh_Cells"],
        results[2]["Mesh_Cells"],
    )
    gci_holdup = compute_asme_gci(
        results[0]["Holdup_CFD"],
        results[1]["Holdup_CFD"],
        results[2]["Holdup_CFD"],
        results[0]["Mesh_Cells"],
        results[1]["Mesh_Cells"],
        results[2]["Mesh_Cells"],
    )

    for i, row in enumerate(results):
        row["Erro_Rel_GradP_Pct"] = (
            0.0 if i == 0 else round(gci_dp["ea_32_pct"] if i == 1 else gci_dp["ea_21_pct"], 2)
        )
        row["GCI_GradP_Pct"] = (
            0.0 if i == 0 else round(gci_dp["gci_med_pct"] if i == 1 else gci_dp["gci_fine_pct"], 2)
        )
        row["GCI_Holdup_Pct"] = (
            0.0 if i == 0 else round(gci_holdup["gci_med_pct"] if i == 1 else gci_holdup["gci_fine_pct"], 2)
        )
        row["GCI_Ordem_p"] = round(gci_dp["p_order"], 3)
        row["GCI_Asymptotic_Ratio"] = round(gci_dp["asymptotic_ratio"], 3)
        row["GCI_Convergence"] = gci_dp["convergence_type"]
        row["GradP_Extrapolado_Pa_m"] = round(gci_dp["phi_ext_21"], 4)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.time() - total_start_time
    mins, secs = divmod(total_elapsed, 60)

    print("\n" + "=" * 90)
    print(" RESULTADOS — INDEPENDÊNCIA DE MALHA (ASME GCI)")
    print("=" * 90)
    print(df[["Malha", "Mesh_Cells", "yPlus_Avg", "U_Avg_m_s", "U_Max_m_s",
              "Grad_P_CFD_Pa_m", "Holdup_CFD", "Erro_Rel_GradP_Pct", "GCI_GradP_Pct"]].to_string(index=False))
    print("-" * 90)
    print(f"GCI Fina (Grad P): {gci_dp['gci_fine_pct']:.2f}% | Ordem aparente p = {gci_dp['p_order']:.2f} "
          f"| Razão Assintótica = {gci_dp['asymptotic_ratio']:.3f} | Tipo: {gci_dp['convergence_type']}")
    print(f"Valor extrapolado (Richardson): Grad_P = {gci_dp['phi_ext_21']:.2f} Pa/m")
    print(f"Tempo total: {int(mins)}m {secs:.1f}s | Planilha salva em: {OUTPUT_CSV}")

    # Critério ASME: se GCI da malha fina < 5%, a malha Média (1.000) já é suficiente
    recommended_name = "Média" if gci_dp["gci_fine_pct"] < 5.0 else "Fina"
    recommended_factor = GRID_FACTORS[recommended_name]
    recommended_row = results[1] if recommended_name == "Média" else results[2]
    print(
        f"\n[+] Malha recomendada (critério ASME GCI < 5%): '{recommended_name}' "
        f"(mesh_factor={recommended_factor:.3f}, {recommended_row['Mesh_Cells']} células)"
    )
    print("    (Nenhuma alteração automática foi feita em config/simulation_config.py.)")


if __name__ == "__main__":
    run_grid_study()
