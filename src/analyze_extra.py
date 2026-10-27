"""Additional analyses: two-stage contribution, prescription structure,
reweighted comparisons, solver benchmark, KKT bounds, latent-trait sensitivity."""
import json, glob, os
import numpy as np, pandas as pd
from analyze import load, STRATS, NAMES, T

from config import OUT_DIR
OUT = str(OUT_DIR)
B = f"{OUT}/bench"


def _bench_records(tag):
    """Benchmark records of one solver: directory written by bench.py, or the archive."""
    files = glob.glob(f"{B}/{tag}/*.json")
    if files:
        return {json.load(open(f))["pid"]: json.load(open(f)) for f in files}
    import gzip
    path = f"{OUT}/bench_results.jsonl.gz"
    if not os.path.exists(path):
        return {}
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tag") == tag:
                out[r["pid"]] = r
    return out


def _latent_records(v):
    files = glob.glob(f"{OUT}/sens_latent/{v}/[0-9]*.json")
    if files:
        return [json.load(open(f)) for f in files]
    import gzip
    path = f"{OUT}/sens_latent_results.jsonl.gz"
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt") as fh:
        return [r for r in map(json.loads, fh) if r.get("variant") == v]


def stage_contrib(R):
    J1, J2, Jh, rule = [], [], [], []
    for r in R:
        tr = np.array([np.nan if v is None else v for v in r["search"]["trace"]], float)
        k1 = r["search"]["k1"]
        J1.append(np.nanmax(tr[:k1])); J2.append(r["proposed"]["J"]); Jh.append(r["heuristic"]["J"])
        rule.append(np.nanmax(tr[:3]))
    J1, J2, Jh, rule = map(np.array, (J1, J2, Jh, rule))
    return dict(stage1_mean=J1.mean(), final_mean=J2.mean(), heur_mean=Jh.mean(),
                stage2_gain_mean=(J2 - J1).mean(), stage2_gain_rel_pct=100 * (J2 - J1).mean() / J1.mean(),
                pct_improved_stage2=100 * np.mean(J2 - J1 > 1e-4),
                stage1_eq_rule_pct=100 * np.mean(np.abs(J1 - rule) <= 1e-9),
                final_eq_rule_pct=100 * np.mean(np.abs(J2 - rule) <= 1e-9))


def structure(R):
    out = {}
    for s in ["proposed", "heuristic", "population_average"]:
        d = np.array([r[s]["d"] for r in R]); al = np.array([r[s]["alpha"] for r in R])
        dmax = np.array([r["dmax"] for r in R]) if "dmax" in R[0] else None
        out[s] = dict(mean_d=d.mean(), mean_alpha=al.mean(), alpha_first4=al[:, :4].mean(), alpha_last4=al[:, -4:].mean(),
                      alpha_sd_within=np.mean(al.std(1)), d_first=d[:, 0].mean(), d_last=d[:, -1].mean())
    return out


def reweighted(R):
    w = np.load(f"{OUT}/cohort_diag/weights.npy")
    P = pd.read_csv(f"{OUT}/participants.csv")
    wmap = dict(zip(P.pid, w))
    ww = np.array([wmap[r["pid"]] for r in R])
    Jp = np.array([r["proposed"]["J"] for r in R])
    rows = []
    for s in STRATS[1:]:
        Jb = np.array([r[s]["J"] for r in R])
        rows.append(dict(baseline=NAMES[s], rel_unweighted=100 * (Jp.mean() / Jb.mean() - 1),
                         rel_weighted=100 * (np.average(Jp, weights=ww) / np.average(Jb, weights=ww) - 1),
                         better_weighted=100 * np.average(Jp - Jb > 1e-9, weights=ww)))
    return pd.DataFrame(rows)


def evals_to_final(trace, tol=1e-4):
    tr = pd.Series([np.nan if v is None else v for v in trace], dtype=float).ffill().values
    fin = tr[-1]
    idx = np.flatnonzero(tr >= fin - tol)
    return int(idx[0]) + 1 if len(idx) else len(tr)


def bench_table(main):
    tags = {"ga_cold": "GA, cold", "tpe_cold": "TPE, cold", "cma_cold": "CMA-ES, cold",
            "ga_warm": "GA, warm", "tpe_warm": "TPE, warm", "cma_warm": "CMA-ES, warm",
            "ga_warm_b1000": "GA, warm", "cma_warm_b1000": "CMA-ES, warm"}
    pids = sorted(_bench_records("cma_warm"))
    Jf = np.array([main[p]["proposed"]["J"] for p in pids]); Jh = np.array([main[p]["heuristic"]["J"] for p in pids])
    rows = []
    def add(name, budget, J, ev, rt):
        J = np.array(J, float)
        rows.append(dict(solver=name, budget=budget, meanJ=np.nanmean(J), gap_to_final_pct=100 * np.nanmean((J - Jf) / Jf),
                         gain_vs_heur_pct=100 * (np.nanmean(J) / Jh.mean() - 1),
                         final_better_pct=100 * np.mean(Jf - J > 1e-4), solver_better_pct=100 * np.mean(J - Jf > 1e-4),
                         median_evals_to_final=np.median(ev), mean_runtime_s=np.mean(rt)))
    for t, nm in tags.items():
        R = _bench_records(t)
        if len(R) < len(pids):
            continue
        add(nm, 1000 if t.endswith("b1000") else 250, [R[p]["J"] for p in pids],
            [evals_to_final(R[p]["trace"]) for p in pids], [R[p]["runtime"] for p in pids])
    # stage one alone and full two-stage, from the main records
    J1, e1, e2, rt = [], [], [], []
    for p in pids:
        s = main[p]["search"]; tr = s["trace"]; k1 = s["k1"]
        J1.append(np.nanmax(np.array([np.nan if v is None else v for v in tr[:k1]], float)))
        e1.append(evals_to_final(tr[:k1])); e2.append(evals_to_final(tr)); rt.append(s["runtime"])
    add("Stage one (surrogate-assisted), warm", "<=250", J1, e1, [np.nan] * len(pids))
    add("Two-stage (proposed), warm", 1000, Jf, e2, rt)
    add("Heuristic progression", "--", Jh, [np.nan] * len(pids), [np.nan] * len(pids))
    return pd.DataFrame(rows), pids


def kkt_table(main):
    rows = []
    for r in sorted(_bench_records("kkt").values(), key=lambda z: z["pid"]):
        p = r["pid"]
        Jf = main[p]["proposed"]["J"]
        rows.append(dict(pid=p, J_two_stage=Jf, J_kkt_incumbent=r.get("J"), primal_kkt=r.get("primal_bound"),
                         dual_bound=r["dual_bound"], gap_pct=100 * (r["dual_bound"] - Jf) / Jf,
                         J_heur=main[p]["heuristic"]["J"], status=r["status"], runtime=r["runtime"],
                         max_mult=r.get("max_abs_multiplier"), max_slack=r.get("max_slack")))
    return pd.DataFrame(rows)


def latent_table():
    rows = []
    for v in ["slope0", "slope1.5", "noise0.5", "noise2", "mapping0.5"]:
        R = _latent_records(v)
        if not R:
            continue
        Jp = np.array([r["proposed"]["J"] for r in R])
        row = dict(variant=v, n=len(R), J=Jp.mean())
        for s in STRATS[1:]:
            Jb = np.array([r[s]["J"] for r in R])
            row[s] = 100 * (Jp.mean() / Jb.mean() - 1)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    R = load("main"); main = {r["pid"]: r for r in R}
    print("N", len(R))
    print(stage_contrib(R))
    print(json.dumps(structure(R), indent=1, default=float))
    print(reweighted(R).round(3))
    bt, pids = bench_table(main); print(bt.round(4))
    print(kkt_table(main).round(5))
    print(latent_table().round(3))
