import os
import config  # noqa: F401  (sets single-threaded numerical libraries)
"""Experiment runner (checkpointed, resumable). Usage:
  python run.py popavg                -> optimize the population-average prescription
  python run.py main  <start> <end>   -> proposed + baselines + robustness for pids in [start,end)
  python run.py ablation <start> <end> -> four ablations for subset pids in [start,end)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from model import (T, Search, search_hybrid, weights, repair, evaluate, Stats, average_participant, COEF_KEYS,
                   PARAM_KEYS, iR, iE, iD, ia, i_f, im, ih)
from cohort import CONST

OUT = str(config.OUT_DIR)
REFIT = 5
W_STAG = 50
BUDGET = 1000
P = pd.read_csv(f"{OUT}/participants.csv")
W = weights()


def pdict(i):
    return P.iloc[i].to_dict()


def seed_of(pid):
    return 42 + 1009 * pid


def pack(p, x, o):
    y = o["sol"]["y"] if o.get("ok") else None
    rec = {k: (o.get(k) if o.get("ok") else None) for k in ["J", "m", "h", "mean_a", "mean_f", "peak_f", "final_a"]}
    rec["ok"] = bool(o.get("ok")); rec["feasible"] = bool(o.get("feasible", False))
    rec["d"] = np.round(x[0], 5).tolist(); rec["alpha"] = np.round(x[1], 5).tolist()
    rec["q"] = np.asarray(x[2]).astype(int).tolist(); rec["o"] = np.asarray(x[3]).astype(int).tolist()
    if y is not None:
        rec["a"] = np.round(y[ia], 5).tolist(); rec["f"] = np.round(y[i_f], 5).tolist()
        rec["uR"] = np.round(y[iR], 5).tolist(); rec["uE"] = np.round(y[iE], 5).tolist()
        rec["flagged"] = o["sol"]["flagged"]; rec["tie"] = o["sol"]["tie"]; rec["dJ"] = o["sol"]["dJ"]
    return rec


# ---------------------------------------------------------------- scenarios
def scenario_params(p, sid, pid):
    p = dict(p)
    rng = np.random.default_rng(42 + 1009 * pid + 1000003 * sid)
    if sid in (1,):
        s = np.sqrt(np.log(1 + 0.05 ** 2))
        for k in COEF_KEYS:
            p[k] = p[k] * rng.lognormal(-s * s / 2, s)
    if sid in (2, 6):
        eta = np.full(T, p["eta"]); eta[4:6] *= 0.75; p["etaseq"] = eta
    if sid in (3, 6):
        p["rho"] = 0.8 * p["rho"]
    if sid in (4, 6):
        for k in ["betR", "betE", "betf", "gamR", "gamE", "gamo", "gamf"]:
            p[k] = p[k] * rng.uniform(0.85, 1.15)
    if sid == 5:
        p["fbar"] = 0.85 * p["fbar"]
    return p


def s7(p, x, o, lam_nl=0.30, f0=0.30):
    """Evaluation-only nonlinear adherence drop: forward simulation of the
    nominal realized execution, capped by the perturbed adherence state."""
    if not o.get("ok"):
        return dict(ok=False, feasible=False)
    y = o["sol"]["y"]
    d, al, q, oo = x
    r, e = al * d, (1 - al) * d
    dprev = np.concatenate([[p["d0"]], d[:-1]])
    prog = (d + CONST["eps_d"]) / (dprev + CONST["eps_d"])
    Dl = np.abs(d - dprev)
    a = np.zeros(T + 1); f = np.zeros(T + 1); m = np.zeros(T + 1); h = np.zeros(T + 1)
    a[0], h[0] = p["eta"], p["sigma"]
    for t in range(T):
        uR = min(y[iR[t]], a[t]); uE = min(y[iE[t]], a[t])
        an = a[t] + p["lam_eta"] * (p["eta"] - a[t]) - p["lam_f"] * f[t] - p["lam_p"] * Dl[t]
        an -= lam_nl * max(f[t] - f0, 0) ** 2 * max(prog[t] - 1, 0)
        a[t + 1] = min(max(an, 0), 1)
        f[t + 1] = (1 - p["rho"]) * f[t] + p["kapR"] * uR * r[t] + p["kapE"] * uE * e[t] + p["kapD"] * Dl[t]
        m[t + 1] = m[t] + p["betR"] * uR * r[t] + p["betE"] * uE * e[t] - p["betf"] * f[t]
        h[t + 1] = (h[t] + p["gamR"] * uR * r[t] + p["gamE"] * uE * e[t]
                    + p["gamo"] * q[t] * (2 * oo[t] - 1) * a[t] * d[t] - p["gamf"] * f[t])
    J = (W["m"] * m[T] + W["h"] * h[T] + W["a"] * a[:T].mean() - W["f"] * f[:T].mean()
         - W["d"] * np.sum(np.diff(d) ** 2) - W["al"] * np.sum(np.diff(al) ** 2))
    feas = bool(a[:T].mean() >= p["abar"] - 1e-9 and f.max() <= p["fbar"] + 1e-9)
    return dict(ok=True, J=float(J), m=float(m[T]), h=float(h[T]), mean_a=float(a[:T].mean()),
                mean_f=float(f[:T].mean()), feasible=feas)


# ---------------------------------------------------------------- baselines
def rule_seeds(p):
    """Rule-based warm-start prescriptions (fixed equal-dose, fixed sequence, heuristic)."""
    b = baselines(p, None, None)
    return [b["fixed_equal"], b["fixed_sequence"], b["heuristic"]]

def baselines(p, pid, xpop):
    d0 = p["d0"]
    out = {}
    out["fixed_equal"] = repair(p, (np.full(T, d0), np.full(T, .50), np.ones(T), np.ones(T)))
    out["fixed_sequence"] = repair(p, (np.full(T, d0), np.full(T, .55), np.ones(T), np.zeros(T)))
    dh, prev = np.zeros(T), d0
    for t in range(T):
        prev = min(1.05 * prev, p["dmax"]); dh[t] = prev
    out["heuristic"] = repair(p, (dh, np.full(T, .50), np.ones(T), np.ones(T)))
    if xpop is not None:
        out["population_average"] = repair(p, xpop)
    return out


def random_feasible(p, pid, K=BUDGET):
    rng = np.random.default_rng(seed_of(pid) + 17)
    st = Stats()
    best, bestJ, bestx, besto = None, -np.inf, None, None
    fallback = (None, -np.inf, None)
    pats = np.array([[0, 0], [1, 0], [1, 1]])
    for _ in range(K):
        qo = pats[rng.integers(0, 3, T)]
        x = repair(p, (rng.uniform(p["dmin"], p["dmax"], T), rng.uniform(.3, .7, T), qo[:, 0], qo[:, 1]))
        o = evaluate(p, x, W, st)
        if not o["ok"]:
            continue
        if o["feasible"] and o["J"] > bestJ:
            bestJ, bestx, besto = o["J"], x, o
        if o["J"] > fallback[1]:
            fallback = (x, o["J"], o)
    if bestx is None:
        bestx, _, besto = fallback
    return bestx, besto


def run_main(i):
    pid = int(P.iloc[i]["pid"])
    path = f"{OUT}/main/{pid:04d}.json"
    if os.path.exists(path):
        return pid
    p = pdict(i)
    t0 = time.perf_counter()
    S = search_hybrid(p, W, seed=seed_of(pid), budget=BUDGET, refit_every=REFIT, w_stag=W_STAG,
                      seeds=rule_seeds(p), struct_moves=True)
    rec = {"pid": pid}
    if S.best is not None:
        xb = S.best
        ob = S.best_out
    else:  # no realized-feasible design found: keep best QP-successful design
        Jv = np.array(S.Jv); j = int(np.nanargmax(Jv))
        from model import decode
        xb = repair(p, decode(p, S.Z[j])); ob = evaluate(p, xb, W)
    dJ = np.abs(np.array(S.stats.dJ)) if S.stats.dJ else np.zeros(1)
    rec["proposed"] = pack(p, xb, ob)
    rec["search"] = dict(k=S.k, best_eval=S.best_eval, stop=S.stop, restarts=S.restarts, runtime=S.runtime,
                         k1=S.k1, k2=S.k2, stop2=getattr(S, "stop2", None),
                         qp=S.stats.as_dict(), qp_time_median=float(np.median(S.stats.qp_time)),
                         dJ_max=float(dJ.max()), dJ_median=float(np.median(dJ)),
                         dJ_gt_1e6=int((dJ > 1e-6).sum()), trace=[None if not np.isfinite(v) else round(v, 6) for v in S.trace])
    xpop = tuple(np.array(v) for v in json.load(open(f"{OUT}/popavg.json"))["x"])
    strat = {"proposed": xb}
    strat.update(baselines(p, pid, xpop))
    outs = {"proposed": ob}
    for k in ["fixed_equal", "fixed_sequence", "heuristic", "population_average"]:
        outs[k] = evaluate(p, strat[k], W)
        rec[k] = pack(p, strat[k], outs[k])
    xr, orr = random_feasible(p, pid)
    strat["random_feasible"], outs["random_feasible"] = xr, orr
    rec["random_feasible"] = pack(p, xr, orr)
    rob = {}
    for sid in range(1, 7):
        ps = scenario_params(p, sid, pid)
        for k in ["proposed", "population_average", "heuristic"]:
            o = evaluate(ps, strat[k], W)
            rob[f"S{sid}|{k}"] = dict(J=o["J"] if o["ok"] else None, feasible=bool(o.get("feasible", False)), ok=o["ok"])
    for k in strat:
        o = s7(p, strat[k], outs[k])
        rob[f"S7|{k}"] = o
    rec["robustness"] = rob
    rec["wall"] = time.perf_counter() - t0
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w"), default=lambda v: v.item() if hasattr(v, "item") else str(v))
    os.replace(tmp, path)
    return pid


# ---------------------------------------------------------------- ablations
AVG = {k: float(P[k].mean()) for k in COEF_KEYS + ["eta", "rho", "sigma"]}


def run_ablation(i):
    pid = int(P.iloc[i]["pid"])
    path = f"{OUT}/ablation/{pid:04d}.json"
    if os.path.exists(path):
        return pid
    p = pdict(i)
    rec = {"pid": pid}
    variants = {
        "without_participant_traits": (dict(p, **AVG), W, None),
        "without_sequence_optimization": (p, W, (np.zeros(T, int), np.zeros(T, int))),
        "without_fatigue_coupling": (dict(p, lam_f=0.0, betf=0.0, gamf=0.0), W, None),
        "without_regularization": (p, weights(noreg=True), None),
    }
    for name, (pv, wv, fixed) in variants.items():
        S = search_hybrid(pv, wv, seed=seed_of(pid), budget=BUDGET, refit_every=REFIT, w_stag=W_STAG,
                          binaries_fixed=fixed, seeds=rule_seeds(pv), struct_moves=True)
        if S.best is None:
            from model import decode
            Jv = np.array(S.Jv); j = int(np.nanargmax(Jv)); xb = repair(pv, decode(pv, S.Z[j]))
        else:
            xb = S.best
        o = evaluate(p, xb, W)  # evaluated under the FULL model and full objective
        r = pack(p, xb, o)
        r["search_k"] = S.k
        r["mean_d"] = float(np.mean(xb[0])); r["mean_alpha"] = float(np.mean(xb[1]))
        rec[name] = r
    json.dump(rec, open(path + ".tmp", "w"), default=lambda v: v.item() if hasattr(v, "item") else str(v)); os.replace(path + ".tmp", path)
    return pid


def run_popavg():
    pa = average_participant(P)
    S = search_hybrid(pa, W, seed=42, budget=BUDGET, refit_every=REFIT, w_stag=W_STAG, seeds=rule_seeds(pa),
                      struct_moves=True)
    json.dump({"x": [np.asarray(v).tolist() for v in S.best], "J": S.bestJ, "k": S.k, "stop": S.stop,
               "p": pa}, open(f"{OUT}/popavg.json", "w"), indent=1)
    print("popavg", S.bestJ, S.k, S.stop)


if __name__ == "__main__":
    mode = sys.argv[1]
    os.makedirs(f"{OUT}/main", exist_ok=True); os.makedirs(f"{OUT}/ablation", exist_ok=True)
    if mode == "popavg":
        run_popavg(); sys.exit()
    a, b = int(sys.argv[2]), int(sys.argv[3])
    if mode == "ablation":
        idx = json.load(open(f"{OUT}/ablation_subset.json"))[a:b]
        fn = run_ablation
    else:
        idx = list(range(a, b)); fn = run_main
    t0 = time.time()
    with Pool(config.N_WORKERS) as pool:
        for n, pid in enumerate(pool.imap_unordered(fn, idx, chunksize=1)):
            if n % 25 == 0:
                print(f"{mode} done {n + 1}/{len(idx)} pid={pid} elapsed={time.time() - t0:.0f}s", flush=True)
    print("FINISHED", mode, a, b, time.time() - t0, flush=True)
