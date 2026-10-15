#!/usr/bin/env bash
# Full reproduction of the reported analysis. Set N_WORKERS to the number of CPU cores to use.
# Steps 3-5 are computationally intensive (about 11 CPU-hours in total on 2.1 GHz Xeon cores).
set -euo pipefail
cd "$(dirname "$0")/src"
python cohort.py                      # 1. calibration cohort, synthetic cohort, participant parameters
python select_cases.py                #    representative case-study participants
python run.py popavg                  # 2. population-average prescription
python run.py main 0 3000             # 3. proposed framework, baselines, robustness (S1-S7)
python - <<'PY'
import json, numpy as np, pandas as pd
from config import OUT_DIR
P = pd.read_csv(OUT_DIR / "participants.csv")
json.dump(sorted(np.random.default_rng(42).choice(len(P), 500, replace=False).tolist()),
          open(OUT_DIR / "ablation_subset.json", "w"))
PY
python run.py ablation 0 500          # 4. ablations on the 500-participant subset
python sensitivity.py                 # 5. optimistic-selection sensitivity analysis
python slack_check.py                 #    epigraph slack and reduced-Hessian tolerance checks
python consolidate.py                 # 6. pack per-participant records
python make_tables.py                 # 7. all tables (outputs/tables)
python figures.py                     # 8. Figures 2-6 (figures/)
