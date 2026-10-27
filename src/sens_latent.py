import os
import config  # noqa: F401  (single-threaded numerical libraries)
"""Latent-trait generation sensitivity: regenerate participant parameters under
alternative trait rules (phenotypes fixed), re-optimize the population-average
prescription, and rerun the two-stage framework and all comparators on the
benchmark participants. Usage: python sens_latent.py <variant> <n_participants>"""
import sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from cohort_variants import build_participants_v
from model import search_hybrid, evaluate, average_participant, repair
import run
from run import W, seed_of, rule_seeds, baselines, random_feasible, pack, REFIT, W_STAG, BUDGET

OUT = str(config.OUT_DIR)
VARIANTS = {"slope0": dict(slope=0.0), "slope1.5": dict(slope=1.5), "noise0.5": dict(noise=0.5),
            "noise2": dict(noise=2.0), "mapping0.5": dict(mapping=0.5)}
cal = pd.read_csv(f"{OUT}/calibration_cohort.csv")
P0 = pd.read_csv(f"{OUT}/participants.csv")
SYN = P0[["age", "height", "weight", "waist", "activity", "female", "bmi", "whtr"]]


def variant_participants(v):
    Pv, _ = build_participants_v(cal, SYN, **VARIANTS[v])
    return Pv


def popavg(v, Pv):
    path = f"{OUT}/sens_latent/{v}/popavg.json"
    if os.path.exists(path):
        return tuple(np.array(a) for a in json.load(open(path))["x"])
    pa = average_participant(Pv)
    S = search_hybrid(pa, W, seed=42, budget=BUDGET, refit_every=REFIT, w_stag=W_STAG, seeds=rule_seeds(pa), struct_moves=True)
    json.dump({"x": [np.asarray(a).tolist() for a in S.best], "J": S.bestJ}, open(path, "w"))
    return S.best


def job(args):
    v, i, xpop = args
    Pv = variant_participants(v)
    p = Pv.iloc[i].to_dict(); pid = int(p["pid"])
    path = f"{OUT}/sens_latent/{v}/{pid:04d}.json"
    if os.path.exists(path):
        return pid
    S = search_hybrid(p, W, seed=seed_of(pid), budget=BUDGET, refit_every=REFIT, w_stag=W_STAG,
                      seeds=rule_seeds(p), struct_moves=True)
    rec = {"pid": pid, "variant": v, "proposed": pack(p, S.best, S.best_out)}
    b = baselines(p, pid, xpop)
    for k in ["fixed_equal", "fixed_sequence", "heuristic", "population_average"]:
        rec[k] = pack(p, b[k], evaluate(p, b[k], W))
    xr, orr = random_feasible(p, pid)
    rec["random_feasible"] = pack(p, xr, orr)
    rec["dmax"] = p["dmax"]
    json.dump(rec, open(path + ".tmp", "w"), default=lambda z: z.item() if hasattr(z, "item") else str(z))
    os.replace(path + ".tmp", path)
    return pid


if __name__ == "__main__":
    from bench import bench_ids
    v = sys.argv[1]; n = int(sys.argv[2])
    os.makedirs(f"{OUT}/sens_latent/{v}", exist_ok=True)
    Pv = variant_participants(v)
    xpop = popavg(v, Pv)
    idx = bench_ids()[:n]
    t0 = time.time()
    with Pool(config.N_WORKERS) as pool:
        for k, pid in enumerate(pool.imap_unordered(job, [(v, i, xpop) for i in idx], chunksize=1)):
            if k % 20 == 0:
                print(v, k + 1, pid, round(time.time() - t0), flush=True)
    print("FINISHED", v, time.time() - t0, flush=True)
