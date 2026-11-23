import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.simulation_config import SimulationConfig
import scripts.generate_mesh as gm
from scripts.post_process import (
    EXPERIMENTAL_DATA,
    ModeloFasesSeparadasChisholm,
    SAMPLING_END_PCT,
    SAMPLING_START_PCT,
    extract_cfd_gradient_and_holdup,
    extract_yplus,
    get_mesh_cells,
)
from scripts.setup_0 import setup_zero_files
from scripts.setup_constant import setup_constant_files
from scripts.setup_system import setup_system_files


# Bateria de testes variando o coeficiente alpha do multiphaseStabilizedTurbulence
ALPHA_CASES = [
    (0.05, "Quase sem amortecimento interfacial (baseline antigo)"),
    (1.36, "Padrao da literatura (Larsen & Fuhrman, 2018)"),
    (2.50, "Amortecimento interfacial intermediario"),
    (4.00, "Amortecimento interfacial forte + limitNut c=35"),
]

END_TIME_S = 3.0
WALL_EXPANSION_RATIO = 8.0
LAMBDA2_MST = 0.05
LIMIT_NUT_C = 35
C_ALPHA = 1.0
OUTPUT_CSV = ROOT_DIR / "results_alpha_sensitivity.csv"


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
    subprocess.run(
        f"{OF_ENV} {cmd}",
        shell=True,
        executable="/bin/bash",
        cwd=cwd,
        check=True,
        env=os.environ.copy(),
    )


def patch_fv_options_alpha(case_dir: Path, alpha_val: float, lambda2_val: float, limit_c: int):
    """Atualiza alpha, lambda2 e limitNut c diretamente no system/fvOptions do caso."""
    fv_opt_path = case_dir / "system" / "fvOptions"
    text = fv_opt_path.read_text(encoding="utf-8")
    text = re.sub(r"(\balpha\s+)[\d\.eE+-]+;", rf"\g<1>{alpha_val};", text)
    text = re.sub(r"(\blambda2\s+)[\d\.eE+-]+;", rf"\g<1>{lambda2_val};", text)
    text = re.sub(r"(\bc\s+)[\d\.eE+-]+;", rf"\g<1>{limit_c};", text)
    fv_opt_path.write_text(text, encoding="utf-8")


def main():
    cfg = SimulationConfig().model_copy(update={"mesh_factor": 1.0, "end_time": END_TIME_S})
    z_start = cfg.length * (SAMPLING_START_PCT / 100.0)
    z_end = cfg.length * (SAMPLING_END_PCT / 100.0)
    num_cores = cfg.num_processors

    exp = EXPERIMENTAL_DATA
    lm = ModeloFasesSeparadasChisholm(
        Q_L=exp["LiqVolFlowrate_m3s"],
        Q_G=exp["GasVolFlowrate_m3s"],
        D=exp["PipeDia_m"],
        theta_deg=getattr(cfg, "inclination_deg", exp["PipeIncli_deg"]),
        rho_L=exp["LiqDensity_kgm3"],
        rho_G=exp["GasDensity_kgm3"],
        mu_L=exp["LiqViscosity_Pas"],
        mu_G=exp["GasViscosity_Pas"],
        p_Pa=exp["SystPressure_Pa"],
    )
    dp_exp = exp["PressureDrop_Exp_Pa_m"]
    holdup_exp = exp["LiqHoldUp_Exp"]

    rows = []
    print("=" * 85)
    print(f" BATERIA DE SENSIBILIDADE: alpha (multiphaseStabilizedTurbulence) | end_time={END_TIME_S}s")
    print("=" * 85)

    for alpha_val, note in ALPHA_CASES:
        tag = f"alpha_{alpha_val:.2f}".replace(".", "_")
        case_dir = ROOT_DIR / "runs" / tag
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[+] Iniciando {tag} (alpha={alpha_val}) — {note}")
        t0 = time.time()

        setup_system_files(cfg, case_dir)
        setup_constant_files(cfg, case_dir, model_name="kOmegaSST")
        setup_zero_files(cfg, case_dir)
        gm.generate_blockmesh_dict(
            cfg, case_dir / "system" / "blockMeshDict", wall_expansion_ratio=WALL_EXPANSION_RATIO
        )
        patch_fv_options_alpha(case_dir, alpha_val, LAMBDA2_MST, LIMIT_NUT_C)

        status = "OK"
        grad_p = 0.0
        holdup = 0.0
        yplus = {"min": 0.0, "max": 0.0, "avg": 0.0, "status": "N/A"}
        n_cells = 0

        try:
            run_cmd("blockMesh > log.blockMesh 2>&1", case_dir)
            run_cmd("decomposePar -force > log.decomposePar 2>&1", case_dir)
            run_cmd(f"mpirun -np {num_cores} interFoam -parallel > log.interFoam 2>&1", case_dir)
            run_cmd("reconstructPar > log.reconstructPar 2>&1", case_dir)

            n_cells = get_mesh_cells(case_dir)
            cfd = extract_cfd_gradient_and_holdup(case_dir, z_start, z_end, num_cores)
            yplus = extract_yplus(case_dir, num_cores)
            grad_p = cfd["grad_p_CFD_Pa_m"]
            holdup = cfd["holdup_CFD_avg"]
        except subprocess.CalledProcessError as exc:
            status = f"ERRO (exit {exc.returncode})"
            print(f"  [!] Falha em {tag}: {status}")

        elapsed = time.time() - t0
        err_exp = abs(grad_p - dp_exp) / dp_exp * 100.0 if dp_exp > 0 and status == "OK" else 0.0
        err_lm = abs(grad_p - lm.dp_dz_total_incomp) / lm.dp_dz_total_incomp * 100.0 if status == "OK" else 0.0
        err_h = abs(holdup - holdup_exp) / holdup_exp * 100.0 if holdup_exp > 0 and status == "OK" else 0.0

        rows.append({
            "Model": "kOmegaSST",
            "Alpha_MST": alpha_val,
            "Lambda2_MST": LAMBDA2_MST,
            "LimitNut_c": LIMIT_NUT_C,
            "cAlpha": C_ALPHA,
            "Wall_Exp_Ratio": WALL_EXPANSION_RATIO,
            "End_Time_s": END_TIME_S,
            "Grad_P_Exp_Pa_m": round(dp_exp, 4),
            "Grad_P_Martinelli_Pa_m": round(lm.dp_dz_total_incomp, 2),
            "Grad_P_CFD_Pa_m": round(grad_p, 4),
            "Error_CFD_vs_Exp_Pct": round(err_exp, 2),
            "Error_CFD_vs_LM_Pct": round(err_lm, 2),
            "Holdup_Exp": holdup_exp,
            "Holdup_Martinelli": round(lm.y_L, 2),
            "Holdup_CFD": round(holdup, 4),
            "Error_Holdup_vs_Exp_Pct": round(err_h, 2),
            "yPlus_Min": round(yplus["min"], 2),
            "yPlus_Max": round(yplus["max"], 2),
            "yPlus_Avg": round(yplus["avg"], 2),
            "yPlus_Status": yplus["status"],
            "Mesh_Cells": n_cells,
            "Cores": num_cores,
            "Execution_Time_s": round(elapsed, 2),
            "Status": status,
            "Observacoes": note,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Salva incrementalmente a cada simulação concluída
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        print(f"  -> Salvo em {OUTPUT_CSV.name}: Grad_P={grad_p:.2f} Pa/m | Holdup={holdup:.4f} | y+={yplus['avg']:.2f}")

    print("\n" + "=" * 85)
    print(f"BATERIA CONCLUÍDA! Planilha salva em: {OUTPUT_CSV}")
    print("=" * 85)


if __name__ == "__main__":
    main()
