"""Write every results table of the manuscript and Supplementary Material to
outputs/tables/*.csv from the stored per-participant records."""
import json
import numpy as np
import pandas as pd
from config import OUT_DIR
from analyze import (load, main_table, paired, computational, robustness, s7_table, sequence_stats,
                     convergence, ablation_table, weekly)

T_DIR = OUT_DIR / "tables"


def tie_table():
    R = json.load(open(OUT_DIR / "tie_sensitivity.json"))
    S = ["proposed", "heuristic", "population_average", "random_feasible", "fixed_equal", "fixed_sequence"]
    rows = []
    for s in S:
        row = {"strategy": s}
        for k in ["J_primary", "J_nominal", "J_tight"]:
            row[f"mean_{k}"] = np.mean([r[s][k] for r in R])
            if s != "proposed":
                d = np.array([r["proposed"][k] - r[s][k] for r in R])
                se = d.std(ddof=1) / np.sqrt(len(d))
                row[f"gain_{k}"] = d.mean(); row[f"gain_{k}_lo"] = d.mean() - 1.96 * se
                row[f"gain_{k}_hi"] = d.mean() + 1.96 * se
                row[f"higher_pct_{k}"] = 100 * np.mean(d > 1e-9); row[f"lower_pct_{k}"] = 100 * np.mean(d < -1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


def margins(R):
    P = pd.read_csv(OUT_DIR / "participants.csv").set_index("pid")
    rows = []
    for s in ["proposed", "heuristic", "population_average"]:
        am = np.array([r[s]["mean_a"] - P.loc[r["pid"], "abar"] for r in R])
        fm = np.array([P.loc[r["pid"], "fbar"] - max(r[s]["f"]) for r in R])
        for name, m in [("adherence", am), ("fatigue", fm)]:
            rows.append(dict(strategy=s, margin=name, minimum=m.min(), p05=np.quantile(m, .05),
                             median=np.median(m), at_bound_pct=100 * np.mean(m <= 1e-6)))
    return pd.DataFrame(rows)


def cohort_fidelity():
    from scipy import stats
    cal = pd.read_csv(OUT_DIR / "calibration_cohort.csv"); syn = pd.read_csv(OUT_DIR / "participants.csv")
    rows = []
    for v in ["age", "female", "height", "weight", "bmi", "waist", "whtr", "activity"]:
        rows.append(dict(variable=v, mean_cal=cal[v].mean(), mean_syn=syn[v].mean(), sd_cal=cal[v].std(),
                         sd_syn=syn[v].std(), median_cal=cal[v].median(), median_syn=syn[v].median(),
                         ks=stats.ks_2samp(cal[v], syn[v]).statistic))
    rows.append(dict(variable="zero_activity_share", mean_cal=(cal.activity == 0).mean(), mean_syn=(syn.activity == 0).mean()))
    return pd.DataFrame(rows)


def cases(R):
    ids = json.load(open(OUT_DIR / "case_ids.json")); P = pd.read_csv(OUT_DIR / "participants.csv").set_index("pid")
    by = {r["pid"]: r for r in R}; rows = []
    for k, pid in sorted(ids.items()):
        x = by[pid]["proposed"]; q = np.array(x["q"]); o = np.array(x["o"])
        if q.sum() == 0:
            pref = "not applicable"
        else:
            m = o[q == 1].mean(); pref = "R->E" if m > 0.5 else ("E->R" if m < 0.5 else "no majority")
        rows.append(dict(case=k, pid=pid, dmax=P.loc[pid, "dmax"], mean_dose=np.mean(x["d"]), resistance_share=np.mean(x["alpha"]),
                         weeks_q1=int(q.sum()), preferred_sequence=pref, final_adherence=x["a"][15], peak_fatigue=max(x["f"]),
                         J=x["J"], J_heuristic=by[pid]["heuristic"]["J"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    T_DIR.mkdir(parents=True, exist_ok=True)
    R = load("main")
    main_table(R).to_csv(T_DIR / "table09_population_benchmark.csv", index=False)
    paired(R).to_csv(T_DIR / "table10_paired_differences.csv", index=False)
    ablation_table(load("ablation")).to_csv(T_DIR / "table11_ablation.csv", index=False)
    tie_table().to_csv(T_DIR / "table12_tie_sensitivity.csv", index=False)
    json.dump(computational(R), open(T_DIR / "table13_computational.json", "w"), indent=1)
    s7_table(R).to_csv(T_DIR / "table14_s7.csv", index=False)
    robustness(R).to_csv(T_DIR / "table15_robustness.csv", index=False)
    cases(R).to_csv(T_DIR / "table16_case_studies.csv", index=False)
    json.dump({k: float(v) if not isinstance(v, int) else v for k, v in sequence_stats(R).items()},
              open(T_DIR / "sequence_counts.json", "w"), indent=1)
    cohort_fidelity().to_csv(T_DIR / "tableS_cohort_fidelity.csv", index=False)
    margins(R).to_csv(T_DIR / "tableS_constraint_margins.csv", index=False)
    q, M, _ = convergence(R)
    pd.DataFrame({"evaluation": np.arange(1, q.shape[1] + 1), "q25": q[0], "median": q[1], "q75": q[2],
                  "running": [(np.array([r["search"]["k"] for r in R]) >= e).sum() for e in range(1, q.shape[1] + 1)]}
                 ).to_csv(T_DIR / "figure4_convergence.csv", index=False)
    for s in ["proposed", "population_average", "heuristic"]:
        d, a, f = weekly(R, s)
        pd.DataFrame({"week": np.arange(1, len(d) + 1), "dose": d, "adherence": a, "fatigue": f}).to_csv(
            T_DIR / f"figures2_3_weekly_{s}.csv", index=False)
    print("tables written to", T_DIR)
