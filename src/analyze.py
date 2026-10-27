"""Aggregate experiment outputs into manuscript tables (CSV + printed)."""
import glob, json, sys
import numpy as np
import pandas as pd
from cohort import CONST

from config import OUT_DIR
OUT = str(OUT_DIR)
T = CONST["T"]
STRATS = ["proposed", "population_average", "heuristic", "fixed_equal", "fixed_sequence", "random_feasible"]
NAMES = {"proposed": "Proposed bilevel framework", "population_average": "Population-average prescription",
         "heuristic": "Heuristic progression", "fixed_equal": "Fixed equal-dose concurrent",
         "fixed_sequence": "Fixed sequence policy", "random_feasible": "Random feasible allocation"}


def load(kind="main"):
    """Per-participant records: the directory written by run.py if present,
    otherwise the consolidated archive outputs/<kind>_results.jsonl.gz."""
    import gzip, os
    files = sorted(glob.glob(f"{OUT}/{kind}/*.json"))
    if files:
        return [json.load(open(f)) for f in files]
    path = f"{OUT}/{kind}_results.jsonl.gz"
    with gzip.open(path, "rt") as fh:
        return sorted((json.loads(line) for line in fh), key=lambda r: r["pid"])


def main_table(R):
    rows = []
    for s in STRATS:
        v = [r[s] for r in R if r[s]["ok"]]
        rows.append(dict(method=NAMES[s], J=np.mean([x["J"] for x in v]), m=np.mean([x["m"] for x in v]),
                         h=np.mean([x["h"] for x in v]), adherence=np.mean([x["mean_a"] for x in v]),
                         fatigue=np.mean([x["mean_f"] for x in v]),
                         feasible_pct=100 * np.mean([r[s]["feasible"] for r in R]), n=len(v)))
    return pd.DataFrame(rows)


def paired(R):
    rows = []
    Jp = np.array([r["proposed"]["J"] for r in R])
    for s in STRATS[1:]:
        Jb = np.array([r[s]["J"] if r[s]["ok"] else np.nan for r in R])
        d = Jp - Jb; d = d[np.isfinite(d)]
        se = d.std(ddof=1) / np.sqrt(len(d))
        rows.append(dict(baseline=NAMES[s], mean_diff=d.mean(), ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se,
                         rel_pct=100 * d.mean() / np.mean(Jb[np.isfinite(Jb)]),
                         pct_proposed_better=100 * np.mean(d > 1e-9), pct_ties=100 * np.mean(np.abs(d) <= 1e-9)))
    return pd.DataFrame(rows)


def computational(R):
    S = [r["search"] for r in R]
    stop = pd.Series([s["stop"] for s in S]).value_counts()
    qp = pd.DataFrame([s["qp"] for s in S]).sum()
    be = np.array([s["best_eval"] for s in S]); k = np.array([s["k"] for s in S])
    wall = np.array([s["runtime"] for s in S])
    fin = pd.Series([r["proposed"].get("tie", "none") for r in R]).value_counts()
    finflag = int(np.sum([r["proposed"].get("flagged", False) for r in R]))
    out = dict(n=len(R), median_evals_total=float(np.median(k)), mean_evals_total=float(k.mean()),
               median_best_eval=float(np.median(be)), mean_best_eval=float(be.mean()), sd_best_eval=float(be.std(ddof=1)),
               stop=stop.to_dict(), qp=qp.to_dict(),
               median_qp_ms=float(np.median([s["qp_time_median"] for s in S]) * 1000),
               median_search_s=float(np.median(wall)), p95_search_s=float(np.percentile(wall, 95)),
               serial_hours=float(wall.sum() / 3600),
               dJ_max=float(max(s["dJ_max"] for s in S)), dJ_median_of_medians=float(np.median([s["dJ_median"] for s in S])),
               dJ_gt_1e6=int(sum(s["dJ_gt_1e6"] for s in S)),
               final_flagged=finflag, final_tie=fin.to_dict(),
               median_k1=float(np.median([s.get("k1", s["k"]) for s in S])),
               stop2=pd.Series([s.get("stop2") for s in S]).value_counts().to_dict(),
               best_in_stage2_pct=float(100 * np.mean([s["best_eval"] > s.get("k1", s["k"]) for s in S])))
    return out


def convergence(R, n_max=1000):
    M = np.full((len(R), n_max), np.nan)
    for i, r in enumerate(R):
        tr = np.array([np.nan if v is None else v for v in r["search"]["trace"]], float)
        tr = pd.Series(tr).ffill().values
        M[i, :len(tr)] = tr
        M[i, len(tr):] = tr[-1]
    q = np.nanpercentile(M, [25, 50, 75], axis=0)
    final = M[:, -1]
    ratio = M / final[:, None]
    return q, M, np.nanmedian(ratio, axis=0)


def robustness(R):
    rows = []
    for sid in range(0, 7):
        row = {"scenario": f"S{sid}"}
        for s in ["proposed", "population_average", "heuristic"]:
            if sid == 0:
                v = [r[s]["J"] for r in R if r[s]["ok"]]; f = [r[s]["feasible"] for r in R]
            else:
                e = [r["robustness"][f"S{sid}|{s}"] for r in R]
                v = [x["J"] for x in e if x["ok"]]; f = [x["feasible"] for x in e]
            row[f"J_{s}"] = np.mean(v); row[f"feas_{s}"] = 100 * np.mean(f)
        rows.append(row)
    df = pd.DataFrame(rows)
    for s in ["proposed", "population_average", "heuristic"]:
        df[f"ret_{s}"] = 100 * df[f"J_{s}"] / df[f"J_{s}"].iloc[0]
    return df


def s7_table(R):
    rows = []
    for s in STRATS:
        e = [r["robustness"][f"S7|{s}"] for r in R]
        ok = [x for x in e if x["ok"]]
        nomJ = np.mean([r[s]["J"] for r in R if r[s]["ok"]])
        J = np.mean([x["J"] for x in ok])
        rows.append(dict(method=NAMES[s], J=J, retention=100 * J / nomJ, m=np.mean([x["m"] for x in ok]),
                         h=np.mean([x["h"] for x in ok]), adherence=np.mean([x["mean_a"] for x in ok]),
                         feasible_pct=100 * np.mean([x["feasible"] for x in e])))
    return pd.DataFrame(rows)


def sequence_stats(R):
    q = np.array([r["proposed"]["q"] for r in R]); o = np.array([r["proposed"]["o"] for r in R])
    return dict(q0_pct=100 * np.mean(q == 0), q1o1_pct=100 * np.mean((q == 1) & (o == 1)),
                q1o0_pct=100 * np.mean((q == 1) & (o == 0)),
                all_q1=int(np.sum(q.min(1) == 1)), any_q1o0=int(np.sum(((q == 1) & (o == 0)).any(1))),
                n_weeks=int(q.size))


def weekly(R, s):
    d = np.array([r[s]["d"] for r in R]); a = np.array([r[s]["a"] for r in R if r[s]["ok"]])
    f = np.array([r[s]["f"] for r in R if r[s]["ok"]])
    return d.mean(0), a[:, :T].mean(0), f[:, :T].mean(0)


def ablation_table(A):
    variants = ["without_participant_traits", "without_sequence_optimization", "without_fatigue_coupling",
                "without_regularization"]
    full = {r["pid"]: r for r in load("main")}
    rows = []
    pids = [a["pid"] for a in A]
    fr = [full[p]["proposed"] for p in pids if p in full]
    rows.append(dict(variant="Full bilevel framework", J=np.mean([x["J"] for x in fr]), m=np.mean([x["m"] for x in fr]),
                     h=np.mean([x["h"] for x in fr]), adherence=np.mean([x["mean_a"] for x in fr]),
                     fatigue=np.mean([x["mean_f"] for x in fr]), feasible_pct=100 * np.mean([x["feasible"] for x in fr]),
                     mean_d=np.mean([np.mean(x["d"]) for x in fr]), mean_uR=np.mean([np.mean(x["uR"]) for x in fr]),
                     mean_uE=np.mean([np.mean(x["uE"]) for x in fr]), n=len(fr)))
    for v in variants:
        xs = [a[v] for a in A if a[v]["ok"]]
        rows.append(dict(variant=v, J=np.mean([x["J"] for x in xs]), m=np.mean([x["m"] for x in xs]),
                         h=np.mean([x["h"] for x in xs]), adherence=np.mean([x["mean_a"] for x in xs]),
                         fatigue=np.mean([x["mean_f"] for x in xs]),
                         feasible_pct=100 * np.mean([a[v]["feasible"] for a in A]),
                         mean_d=np.mean([x["mean_d"] for x in xs]), mean_uR=np.mean([np.mean(x["uR"]) for x in xs]),
                         mean_uE=np.mean([np.mean(x["uE"]) for x in xs]), n=len(xs)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    R = load("main")
    print("N =", len(R))
    print(main_table(R).round(4))
    print(paired(R).round(4))
    print(json.dumps(computational(R), indent=1))
    print(robustness(R).round(3))
    print(s7_table(R).round(3))
    print(sequence_stats(R))
    q, M, ratio = convergence(R)
    for e in [1, 25, 40, 50, 100, 150, 200, 250]:
        print(e, q[:, e - 1].round(4), round(ratio[e - 1], 4))
    if len(sys.argv) > 1 and sys.argv[1] == "abl":
        print(ablation_table(load("ablation")).round(4))
