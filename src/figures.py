"""Regenerate manuscript figures from computed outputs (dark academic palette,
secondary encoding by marker and line style)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import load, weekly, convergence, robustness, OUT, T

from config import FIG_DIR
FIG = str(FIG_DIR)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 300})
COL = {"proposed": "#000000", "population_average": "#00308F", "heuristic": "#8B0000"}
MRK = {"proposed": "o", "population_average": "s", "heuristic": "^"}
LS = {"proposed": "-", "population_average": "--", "heuristic": "-."}
LAB = {"proposed": "Proposed bilevel", "population_average": "Population-average", "heuristic": "Heuristic progression"}
weeks = np.arange(1, T + 1)


def style(ax):
    ax.grid(True, color="#d9d9d9", lw=0.5)
    ax.set_axisbelow(True)


def fig_dose(R):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for s in COL:
        d = np.array([r[s]["d"] for r in R]).mean(0)
        al = np.array([r[s]["alpha"] for r in R]).mean(0)
        axs[0].plot(weeks, d, LS[s], color=COL[s], marker=MRK[s], ms=5, lw=1.6, label=LAB[s])
        axs[1].plot(weeks, al, LS[s], color=COL[s], marker=MRK[s], ms=5, lw=1.6, label=LAB[s])
    axs[0].set_ylabel("Mean prescribed weekly dose (normalized)")
    axs[1].set_ylabel("Mean resistance share $\\alpha_t$")
    for ax, t in zip(axs, ["(a) Weekly dose", "(b) Resistance share"]):
        ax.set_xlabel("Week"); ax.set_xticks(weeks[::2]); ax.set_xlim(1, T); style(ax)
        ax.set_title(t, loc="left", fontsize=11)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/dose_trajectory.png"); plt.close(fig)


def fig_states(R):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
    for s in COL:
        _, a, f = weekly(R, s)
        axs[0].plot(weeks, a, LS[s], color=COL[s], marker=MRK[s], ms=5, lw=1.6, label=LAB[s])
        axs[1].plot(weeks, f, LS[s], color=COL[s], marker=MRK[s], ms=5, lw=1.6, label=LAB[s])
    axs[0].set_ylabel("Mean adherence state $a_t$"); axs[1].set_ylabel("Mean fatigue state $f_t$")
    for ax, t in zip(axs, ["(a) Adherence", "(b) Fatigue"]):
        ax.set_xlabel("Week"); ax.set_xticks(weeks[::2]); ax.set_xlim(1, T); style(ax)
        ax.set_title(t, loc="left", fontsize=11)
    axs[1].legend(frameon=True, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/state_trajectory.png"); plt.close(fig)


def fig_conv(R):
    q, M, _ = convergence(R)
    ev = np.arange(1, q.shape[1] + 1)
    G = M - M[:, [2]]  # gain over the best of the three rule-based warm-start designs
    g = np.nanpercentile(G, [25, 50, 75, 90], axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.fill_between(ev, q[0], q[2], color="#9fb3d9", alpha=0.6, lw=0, label="Interquartile range")
    ax.plot(ev, q[1], color="#000000", lw=1.8, label="Median")
    ax.set_ylabel("Best-so-far objective $\\mathcal{J}_i$")
    ax.set_title("(a) Absolute objective", loc="left", fontsize=11)
    ax = axs[1]
    ax.fill_between(ev[2:], g[0][2:], g[2][2:], color="#9fb3d9", alpha=0.6, lw=0, label="Interquartile range")
    ax.plot(ev[2:], g[1][2:], color="#000000", lw=1.8, label="Median")
    ax.plot(ev[2:], g[3][2:], ":", color="#8B0000", lw=1.6, label="90th percentile")
    ax.set_ylabel("Gain over best rule-based start")
    ax.set_title("(b) Improvement beyond rule-based designs", loc="left", fontsize=11)
    for ax in axs:
        ax.axvline(40, color="#006400", ls="--", lw=1.0)
        ax.axvline(90, color="#8B0000", ls=":", lw=1.0)
        ax.set_xlabel("Exact upper-level evaluation"); ax.set_xlim(1, q.shape[1]); style(ax)
        ax.legend(frameon=True, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/objective_convergence.png"); plt.close(fig)


def fig_rob(R):
    df = robustness(R)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s in COL:
        ax.plot(x, df[f"ret_{s}"], LS[s], color=COL[s], marker=MRK[s], ms=6, lw=1.6, label=LAB[s])
    ax.set_xticks(x); ax.set_xticklabels(df["scenario"])
    ax.set_xlabel("Scenario"); ax.set_ylabel("Objective retention relative to S0 (%)")
    style(ax); ax.legend(frameon=True, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/robustness_retention.png"); plt.close(fig)


def fig_cases(R):
    ids = json.load(open(f"{OUT}/case_ids.json"))
    byid = {r["pid"]: r for r in R}
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    pat = {(0, 0): ("$(q_t,o_t)=(0,0)$: separate sessions", "#7f7f7f", "x"),
           (1, 0): ("$(1,0)$: E$\\rightarrow$R", "#00308F", "v"),
           (1, 1): ("$(1,1)$: R$\\rightarrow$E", "#8B0000", "^")}
    for ax, (k, pid) in zip(axs.ravel(), sorted(ids.items())):
        r = byid[pid]["proposed"]
        d, al = np.array(r["d"]), np.array(r["alpha"])
        ax.step(weeks, d, where="mid", color="#000000", lw=1.6, label="Weekly dose $d_t$")
        ax.plot(weeks, d, "o", color="#000000", ms=4)
        ax2 = ax.twinx()
        ax2.plot(weeks, al, "--", color="#006400", lw=1.2, label="Resistance share $\\alpha_t$")
        ax2.set_ylim(0.25, 0.75)
        if k in ("B", "D"):
            ax2.set_ylabel("Resistance share $\\alpha_t$", color="#006400")
        lo = min(d) - 0.12
        for t in range(T):
            name, c, m = pat[(r["q"][t], r["o"][t])]
            ax.plot(weeks[t], lo + 0.04, m, color=c, ms=6)
        ax.set_ylim(lo, max(d) + 0.08)
        ax.set_title(f"Case {k}", fontsize=11); style(ax)
        ax.set_xticks([1, 4, 8, 12, 16]); ax.set_xlim(0.5, 16.5)
        if k in ("A", "C"):
            ax.set_ylabel("Weekly dose (normalized)")
        if k in ("C", "D"):
            ax.set_xlabel("Week")
    h = [plt.Line2D([], [], color="#000000", lw=1.6, label="Weekly dose $d_t$"),
         plt.Line2D([], [], color="#006400", ls="--", lw=1.2, label="Resistance share $\\alpha_t$")]
    h += [plt.Line2D([], [], ls="", marker=m, color=c, label=n) for n, c, m in pat.values()]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{FIG}/case_study_dose_profiles.png"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIG, exist_ok=True)
    R = load("main")
    fig_dose(R); fig_states(R); fig_conv(R); fig_rob(R)
    ids = json.load(open(f"{OUT}/case_ids.json"))
    if all(any(r["pid"] == p for r in R) for p in ids.values()):
        fig_cases(R)
    print("figures done", len(R))
