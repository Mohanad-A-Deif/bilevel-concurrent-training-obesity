"""Repository paths and runtime settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NHANES_DIR", ROOT / "data" / "raw"))
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "outputs"))
FIG_DIR = Path(os.environ.get("FIG_DIR", ROOT / "figures"))
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 2))))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
