import os
import config  # noqa: F401  (single-threaded numerical libraries)
"""Single-level KKT reformulation of the optimistic bilevel problem, solved
globally (spatial branch-and-bound) with SCIP under a time limit.

The lower-level QP is convex and satisfies Slater's condition, so its KKT
conditions are necessary and sufficient. Complementarity is modeled with SOS1
constraints. Because the body-composition and fitness states are unconstrained
and do not enter the lower-level objective, their equality multipliers vanish
identically and are omitted. Products of the upper-level allocation with
lower-level execution are introduced through auxiliary variables r_t = alpha_t d_t,
rr_t = r_t^2, ee_t = e_t^2, so the reformulation is a nonconvex mixed-integer QCQP.
Usage: python kkt.py <start> <end> <time_limit_s>"""
import sys, json, time
import numpy as np
from pyscipopt import Model, quicksum
from model import T, repair, evaluate
from cohort import CONST as C
from run import P, W

OUT = str(config.OUT_DIR)
KDIR = f"{OUT}/bench/kkt"


def build(p, start=None):
    M = Model()
    M.hideOutput()
    D = range(T); D1 = range(T + 1)
    dmin, dmax, d0, Dd = p["dmin"], p["dmax"], p["d0"], p["Dd"]
    eps = C["eps_d"]
    # ---------------- upper level
    d = [M.addVar(f"d{t}", lb=dmin, ub=dmax) for t in D]
    al = [M.addVar(f"al{t}", lb=C["alpha_min"], ub=C["alpha_max"]) for t in D]
    q = [M.addVar(f"q{t}", vtype="B") for t in D]
    o = [M.addVar(f"o{t}", vtype="B") for t in D]
    s = [M.addVar(f"s{t}", lb=0, ub=dmax) for t in D]
    sb = [M.addVar(f"sb{t}", vtype="B") for t in D]
    r = [M.addVar(f"r{t}", lb=0, ub=dmax) for t in D]
    rr = [M.addVar(f"rr{t}", lb=0, ub=dmax ** 2) for t in D]
    ee = [M.addVar(f"ee{t}", lb=0, ub=dmax ** 2) for t in D]
    bigM = dmax - dmin + abs(dmax - d0) + 1.0
    for t in D:
        prev = d0 if t == 0 else d[t - 1]
        M.addCons(d[t] - prev <= Dd); M.addCons(prev - d[t] <= Dd)
        M.addCons(d[t] >= (1 - C["delta_minus"]) * (prev + eps) - eps)
        M.addCons(d[t] <= (1 + C["delta_plus"]) * (prev + eps) - eps)
        if t > 0:
            M.addCons(al[t] - al[t - 1] <= C["Delta_alpha"]); M.addCons(al[t - 1] - al[t] <= C["Delta_alpha"])
        M.addCons(o[t] <= q[t])
        M.addCons(al[t] >= C["alpha_c"] * q[t]); M.addCons(al[t] <= 1 - C["alpha_c"] * q[t])
        # s_t = |d_t - d_{t-1}|
        M.addCons(s[t] >= d[t] - prev); M.addCons(s[t] >= prev - d[t])
        M.addCons(s[t] <= d[t] - prev + bigM * sb[t]); M.addCons(s[t] <= prev - d[t] + bigM * (1 - sb[t]))
        M.addCons(r[t] == al[t] * d[t])
        M.addCons(rr[t] == r[t] * r[t])
        M.addCons(ee[t] == (d[t] - r[t]) * (d[t] - r[t]))
    e = [d[t] - r[t] for t in D]
    # ---------------- lower-level primal
    uR = [M.addVar(f"uR{t}", lb=0, ub=1) for t in D]
    uE = [M.addVar(f"uE{t}", lb=0, ub=1) for t in D]
    Dl = [M.addVar(f"Dl{t}", lb=0, ub=10) for t in D]
    a = [M.addVar(f"a{t}", lb=0, ub=1) for t in D1]
    f = [M.addVar(f"f{t}", lb=0, ub=p["fbar"]) for t in D1]
    m = [M.addVar(f"m{t}", lb=-1e3, ub=1e3) for t in D1]
    h = [M.addVar(f"h{t}", lb=-1e3, ub=1e3) for t in D1]
    M.addCons(a[0] == p["eta"]); M.addCons(f[0] == 0); M.addCons(m[0] == 0); M.addCons(h[0] == p["sigma"])
    for t in D:
        M.addCons(a[t + 1] - (1 - p["lam_eta"]) * a[t] + p["lam_f"] * f[t] + p["lam_p"] * Dl[t] == p["lam_eta"] * p["eta"])
        M.addCons(f[t + 1] - (1 - p["rho"]) * f[t] - p["kapR"] * r[t] * uR[t] - p["kapE"] * e[t] * uE[t] - p["kapD"] * Dl[t] == 0)
        M.addCons(m[t + 1] - m[t] - p["betR"] * r[t] * uR[t] - p["betE"] * e[t] * uE[t] + p["betf"] * f[t] == 0)
        M.addCons(h[t + 1] - h[t] - p["gamR"] * r[t] * uR[t] - p["gamE"] * e[t] * uE[t] + p["gamf"] * f[t]
                  - p["gamo"] * (2 * o[t] - q[t]) * d[t] * a[t] == 0)
    # ---------------- lower-level dual and KKT
    BIG = 1e3
    lA = [M.addVar(f"lA{t}", lb=-BIG, ub=BIG) for t in D]
    lF = [M.addVar(f"lF{t}", lb=-BIG, ub=BIG) for t in D]
    la0 = M.addVar("la0", lb=-BIG, ub=BIG); lf0 = M.addVar("lf0", lb=-BIG, ub=BIG)
    mu = {}

    def comp(name, g_slack_expr):
        mv = M.addVar(f"mu_{name}", lb=0, ub=BIG)
        sv = M.addVar(f"sl_{name}", lb=0, ub=10)
        M.addCons(sv == g_slack_expr)
        M.addConsSOS1([mv, sv])
        mu[name] = mv
        return mv

    for t in D:
        comp(f"1_{t}", a[t] - uR[t]); comp(f"2_{t}", uR[t]); comp(f"3_{t}", uE[t]); comp(f"4_{t}", a[t] - uE[t])
        comp(f"5_{t}", Dl[t] - s[t])
    for t in D1:
        comp(f"alo_{t}", a[t]); comp(f"ahi_{t}", 1 - a[t]); comp(f"flo_{t}", f[t]); comp(f"fhi_{t}", p["fbar"] - f[t])
    for t in D:
        M.addCons(p["psiR"] * rr[t] * uR[t] - p["psiR"] * rr[t] - p["kapR"] * r[t] * lF[t] + mu[f"1_{t}"] - mu[f"2_{t}"] == 0)
        M.addCons(p["psiE"] * ee[t] * uE[t] - p["psiE"] * ee[t] - p["kapE"] * e[t] * lF[t] + mu[f"4_{t}"] - mu[f"3_{t}"] == 0)
        M.addCons(p["lam_p"] * lA[t] - p["kapD"] * lF[t] - mu[f"5_{t}"] == 0)
    for t in D1:
        ex = (la0 if t == 0 else lA[t - 1]) - mu[f"alo_{t}"] + mu[f"ahi_{t}"]
        if t < T:
            ex = ex - p["zeta"] - (1 - p["lam_eta"]) * lA[t] - mu[f"1_{t}"] - mu[f"4_{t}"]
        M.addCons(ex == 0)
        ex = (lf0 if t == 0 else lF[t - 1]) - mu[f"flo_{t}"] + mu[f"fhi_{t}"]
        if t < T:
            ex = ex + 2 * p["chi"] * f[t] + p["lam_f"] * lA[t] - (1 - p["rho"]) * lF[t]
        M.addCons(ex == 0)
    # ---------------- upper-level acceptability and objective
    M.addCons(quicksum(a[t] for t in D) >= T * p["abar"])
    J = M.addVar("J", lb=-1e3, ub=1e3)
    M.addCons(J <= W["m"] * m[T] + W["h"] * h[T] + W["a"] / T * quicksum(a[t] for t in D)
              - W["f"] / T * quicksum(f[t] for t in D)
              - W["d"] * quicksum((d[t] - d[t - 1]) * (d[t] - d[t - 1]) for t in range(1, T))
              - W["al"] * quicksum((al[t] - al[t - 1]) * (al[t] - al[t - 1]) for t in range(1, T)))
    M.setObjective(J, "maximize")
    V = dict(d=d, al=al, q=q, o=o)
    return M, V


def solve(i, tlim):
    p = P.iloc[i].to_dict(); pid = int(p["pid"])
    path = f"{KDIR}/{pid:04d}.json"
    if os.path.exists(path):
        return pid
    M, V = build(p)
    M.setParam("limits/time", tlim)
    M.setParam("numerics/feastol", 1e-7)
    M.setParam("randomization/randomseedshift", 42)
    t0 = time.perf_counter()
    M.optimize()
    rt = time.perf_counter() - t0
    rec = dict(pid=pid, status=M.getStatus(), runtime=rt, nsols=M.getNSols(),
               dual_bound=float(M.getDualbound()), gap=float(M.getGap()) if M.getNSols() else None)
    if M.getNSols():
        sol = M.getBestSol()
        x = (np.array([sol[v] for v in V["d"]]), np.array([sol[v] for v in V["al"]]),
             np.round([sol[v] for v in V["q"]]).astype(int), np.round([sol[v] for v in V["o"]]).astype(int))
        rec["J_model"] = float(M.getPrimalbound())
        xr = repair(p, x)
        o = evaluate(p, xr, W)
        rec["J"] = float(o["J"]) if o["ok"] and o["feasible"] else None
        rec["d"] = np.round(xr[0], 5).tolist(); rec["alpha"] = np.round(xr[1], 5).tolist()
        rec["q"] = xr[2].tolist(); rec["o"] = xr[3].tolist()
    json.dump(rec, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return pid


def start_values(p, x, tlim=120):
    """Complete KKT point (primal, dual, auxiliaries) for a fixed upper-level design x."""
    M, V = build(p)
    for t in range(T):
        M.fixVar(V["d"][t], float(x[0][t])); M.fixVar(V["al"][t], float(x[1][t]))
        M.fixVar(V["q"][t], int(x[2][t])); M.fixVar(V["o"][t], int(x[3][t]))
    M.setParam("limits/time", tlim)
    M.optimize()
    if M.getNSols() == 0:
        return None
    sol = M.getBestSol()
    return {v.name: sol[v] for v in M.getVars()}


def solve_from(i, x0, tlim, kdir):
    p = P.iloc[i].to_dict(); pid = int(p["pid"])
    path = f"{kdir}/{pid:04d}.json"
    if os.path.exists(path):
        return pid
    x0 = repair(p, x0)
    sv = start_values(p, x0)
    M, V = build(p)
    rec = dict(pid=pid, start_ok=sv is not None)
    if sv is not None:
        s = M.createPartialSol()
        byname = {v.name: v for v in M.getVars()}
        for k, val in sv.items():
            if k in byname:
                M.setSolVal(s, byname[k], val)
        M.addSol(s)
    M.setParam("limits/time", tlim)
    M.setParam("numerics/feastol", 1e-7)
    M.setParam("randomization/randomseedshift", 42)
    t0 = time.perf_counter(); M.optimize(); rt = time.perf_counter() - t0
    rec.update(status=M.getStatus(), runtime=rt, nsols=M.getNSols(), dual_bound=float(M.getDualbound()),
               primal_bound=float(M.getPrimalbound()) if M.getNSols() else None)
    rec["J_start"] = float(evaluate(p, x0, W)["J"])
    if M.getNSols():
        sol = M.getBestSol()
        x = (np.array([sol[v] for v in V["d"]]), np.array([sol[v] for v in V["al"]]),
             np.round([sol[v] for v in V["q"]]).astype(int), np.round([sol[v] for v in V["o"]]).astype(int))
        xr = repair(p, x); o = evaluate(p, xr, W)
        rec["J"] = float(o["J"]) if o["ok"] and o["feasible"] else None
        rec["d"] = np.round(xr[0], 5).tolist(); rec["alpha"] = np.round(xr[1], 5).tolist()
        rec["q"] = xr[2].tolist(); rec["o"] = xr[3].tolist()
        vals = {v.name: sol[v] for v in M.getVars()}
        rec["max_abs_multiplier"] = float(max(abs(val) for k, val in vals.items() if k.startswith(("mu_", "lA", "lF", "la0", "lf0"))))
        rec["max_slack"] = float(max(val for k, val in vals.items() if k.startswith("sl_")))
    json.dump(rec, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return pid


if __name__ == "__main__":
    from bench import bench_ids
    os.makedirs(KDIR, exist_ok=True)
    a, b, tl = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    for i in bench_ids()[a:b]:
        t0 = time.time(); pid = solve(i, tl); print("kkt", pid, round(time.time() - t0), flush=True)
