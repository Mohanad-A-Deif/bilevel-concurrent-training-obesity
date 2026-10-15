"""Fast checks: cohort flow, lower-level structure, and consistency of the stored results.
Run with:  cd src && python -m pytest ../tests -q"""
import gzip, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import OUT_DIR  # noqa: E402
from cohort import build_calibration, EXPECTED_FLOW  # noqa: E402
from model import build_qp, solve_lower, repair, weights, iD, T  # noqa: E402

P = pd.read_csv(OUT_DIR / "participants.csv")


def test_cohort_flow():
    _, flow = build_calibration()
    assert flow == EXPECTED_FLOW


def test_hessian_psd_and_epigraph_tight():
    p = P.iloc[0].to_dict()
    x = repair(p, (np.linspace(p["d0"], p["dmax"], T), np.full(T, .5), np.ones(T), np.ones(T)))
    Q = build_qp(p, x)
    assert np.all(Q["Pd"] >= 0)
    sol = solve_lower(p, x, weights(), tiebreak=False)
    assert sol["ok"]
    dprev = np.concatenate([[p["d0"]], x[0][:-1]])
    assert np.max(np.abs(sol["y"][iD][:-1] - np.abs(x[0] - dprev)[:-1])) < 1e-6


def test_stored_objectives_consistent():
    w = weights()
    with gzip.open(OUT_DIR / "main_results.jsonl.gz", "rt") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            x = r["proposed"]; d = np.array(x["d"]); al = np.array(x["alpha"])
            J = (w["m"] * x["m"] + w["h"] * x["h"] + w["a"] * x["mean_a"] - w["f"] * x["mean_f"]
                 - w["d"] * np.sum(np.diff(d) ** 2) - w["al"] * np.sum(np.diff(al) ** 2))
            assert abs(J - x["J"]) < 2e-4
            if i > 200:
                break
