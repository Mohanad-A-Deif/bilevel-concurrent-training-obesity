"""Tie-selection sensitivity: re-evaluate the final prescriptions of every
strategy (i) with the primary OSQP solution only and (ii) with a tie band
100 times tighter than the nominal one."""
import os
import config
import json, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
import model
from model import evaluate, weights, T

OUT = str(config.OUT_DIR)
P = pd.read_csv(f"{OUT}/participants.csv").set_index("pid")
W = weights()
STRATS = ["proposed", "heuristic", "population_average", "random_feasible", "fixed_equal", "fixed_sequence"]


def run(r):
    p = P.loc[r["pid"]].to_dict()
    out = {"pid": r["pid"]}
    for s in STRATS:
        x = (np.array(r[s]["d"]), np.array(r[s]["alpha"]), np.array(r[s]["q"]), np.array(r[s]["o"]))
        o1 = evaluate(p, x, W, tiebreak=False)
        model.TIE_REL, model.TIE_ABS = 1e-8, 1e-9
        o2 = evaluate(p, x, W, tiebreak=True)
        model.TIE_REL, model.TIE_ABS = 1e-6, 1e-7
        o3 = evaluate(p, x, W, tiebreak=True)
        out[s] = dict(J_primary=o1["J"], J_tight=o2["J"], J_nominal=o3["J"], J_stored=r[s]["J"],
                      feas_primary=bool(o1["feasible"]), tight_status=o2["sol"]["tie"])
    return out


if __name__ == "__main__":
    from analyze import load
    with Pool(config.N_WORKERS) as pool:
        res = pool.map(run, load("main"), chunksize=20)
    json.dump(res, open(f"{OUT}/tie_sensitivity.json", "w"), default=lambda v: v.item() if hasattr(v, "item") else str(v))
    print("done", len(res))
