import os
import config
import json, glob, numpy as np, pandas as pd
from multiprocessing import Pool
import model
from model import solve_lower, build_qp, weights, iD, ia, i_f, T, reduced_hessian_deficient
P = pd.read_csv(config.OUT_DIR / "participants.csv").set_index("pid"); W = weights()
def run(r):
    p = P.loc[r["pid"]].to_dict(); x = tuple(np.array(r["proposed"][k]) for k in ["d","alpha","q","o"])
    dprev = np.concatenate([[p["d0"]], x[0][:-1]]); lb = np.abs(x[0]-dprev)
    o1 = solve_lower(p, x, W, tiebreak=False); y = o1["y"]
    s1 = y[iD]-lb
    flag, nul = reduced_hessian_deficient(build_qp(p, x), y)
    o2 = solve_lower(p, x, W, tiebreak=True); y2 = o2["y"]; s2 = y2[iD]-lb
    term = dict(aT1=float(y[ia[T]]), fT1=float(y[i_f[T]]), fbar=p["fbar"])
    out = dict(pid=r["pid"], slack_primary_tlT=float(np.abs(s1[:-1]).max()), slack_primary_T=float(s1[-1]),
               slack_tie_tlT=float(np.abs(s2[:-1]).max()), slack_tie_T=float(s2[-1]), flag=bool(flag), nul=int(nul),
               fT1_at_cap=bool(y[i_f[T]] >= p["fbar"]-1e-6), aT1_at0=bool(y[ia[T]] <= 1e-6))
    # tolerance sensitivity of the flag
    Q = build_qp(p, x)
    for rt, at in [(1e-8, 1e-6), (1e-12, 1e-6), (1e-10, 1e-8), (1e-10, 1e-5)]:
        model.RANK_TOL, model.ACT_TOL = rt, at
        out[f"flag_{rt}_{at}"] = bool(reduced_hessian_deficient(Q, y)[0])
    model.RANK_TOL, model.ACT_TOL = 1e-10, 1e-6
    return out
if __name__ == "__main__":
    from analyze import load
    with Pool(config.N_WORKERS) as pool: res = pool.map(run, load("main"), chunksize=25)
    json.dump(res, open(config.OUT_DIR / "slack_check.json", "w"))
    d = pd.DataFrame(res)
    print(d.describe().T[["mean","min","max"]])
    print(d.filter(like="flag").mean())
    print("nul dims", d.nul.value_counts().to_dict())
