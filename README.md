# Bilevel optimization of concurrent resistance–endurance training in obesity

Code, data, and outputs for the in silico study **"Personalized Concurrent Resistance–Endurance Training in Obesity: A Bilevel Framework for Dose and Sequence Optimization"** (Deif, Elhaty, Hafez, and Khishe).

The repository contains every step of the analysis. It calibrates a baseline phenotype space on NHANES 2013–2014 and generates a synthetic cohort of 3,000 adults with obesity. For each participant, it optimizes a 16-week concurrent-training prescription with a bilevel model, compares the result with five baseline strategies, and runs the ablations, robustness scenarios, and numerical sensitivity checks. Every table and figure of the article can be regenerated from the stored outputs in about a minute, or recomputed from the raw data.

> **Scope.** All results are in silico. NHANES is used only to calibrate baseline phenotypes; the training-response dynamics are prespecified modeling assumptions. The outputs are not clinical exercise prescriptions.

## Model in brief

- **Upper level (prescription).** For each week *t = 1..16*, the upper level chooses the total dose *d_t*, the resistance share *α_t*, same-session concurrency *q_t ∈ {0,1}*, and resistance→endurance order *o_t ∈ {0,1}*. It maximizes a weighted objective of terminal body-composition benefit, terminal fitness, mean adherence, and mean fatigue, minus smoothness penalties. Bounds, change limits, and progression limits apply.
- **Lower level (realization).** A participant-specific convex quadratic program returns the executed fractions and the adherence, fatigue, body-composition, and fitness trajectories for a given prescription. It is solved with OSQP.
- **Optimistic selection.** If the active-set reduced Hessian is rank deficient, an ε-optimistic secondary problem (solved with CLARABEL) selects the realization with the largest upper-level objective.
- **Upper-level search.** A random-forest surrogate-assisted mixed-variable heuristic is warm-started from three rule-based prescriptions, with a budget of 250 exact evaluations and a stagnation window of 50.

All parameter values are listed in `src/cohort.py` (`CONST` and the coefficient mappings) and `src/model.py`.

## Repository structure

```
data/raw/                 NHANES 2013-2014 inputs (DEMO_H, BMX_H, PAQ_H as CSV); see data/README.md
src/config.py             paths and number of worker processes (environment variables OUT_DIR, FIG_DIR, NHANES_DIR, N_WORKERS)
src/cohort.py             calibration cohort, Gaussian-copula synthetic cohort, latent traits, participant parameters
src/select_cases.py       selection of the four representative case-study participants
src/model.py              lower-level QP, reduced-Hessian test, optimistic tie-break, surrogate-assisted search
src/run.py                proposed framework, baselines, robustness scenarios S1-S7, ablations (checkpointed)
src/sensitivity.py        re-evaluation with primary-only and tighter tie bands
src/slack_check.py        epigraph slack, terminal-constraint activity, rank/active-set tolerance checks
src/consolidate.py        packs per-participant records into outputs/*_results.jsonl.gz
src/analyze.py            aggregation functions used by the tables and figures
src/make_tables.py        writes all result tables to outputs/tables/
src/figures.py            regenerates Figures 2-6 in figures/
tests/test_repro.py       fast consistency checks
reproduce.sh              complete pipeline from raw data
outputs/                  stored results of the reported run (see below)
figures/                  Figures 2-6 as reported
```

## Installation

Python 3.13 was used. Install the pinned dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Regenerate the tables and figures from the stored outputs

```bash
cd src
python make_tables.py      # -> outputs/tables/*.csv, *.json
python figures.py          # -> figures/*.png
python -m pytest ../tests -q
```

| Article item | Produced by | File |
|---|---|---|
| Table 9 (population benchmark) | `make_tables.py` | `outputs/tables/table09_population_benchmark.csv` |
| Table 10 (paired differences) | `make_tables.py` | `table10_paired_differences.csv` |
| Table 11 (ablations, 500-participant subset) | `make_tables.py` | `table11_ablation.csv` |
| Table 12 (optimistic-selection sensitivity) | `make_tables.py` | `table12_tie_sensitivity.csv` |
| Table 13 (computational performance, tie handling) | `make_tables.py` | `table13_computational.json` |
| Table 14 (S7 stress test) | `make_tables.py` | `table14_s7.csv` |
| Table 15 (robustness S0-S6) | `make_tables.py` | `table15_robustness.csv` |
| Table 16 (case studies) | `make_tables.py` | `table16_case_studies.csv` |
| (q, o) counts | `make_tables.py` | `sequence_counts.json` |
| Supplementary cohort agreement and constraint margins | `make_tables.py` | `tableS_cohort_fidelity.csv`, `tableS_constraint_margins.csv` |
| Figures 2-3 data | `make_tables.py` | `figures2_3_weekly_*.csv` |
| Figure 4 data | `make_tables.py` | `figure4_convergence.csv` |
| Figures 2-6 | `figures.py` | `figures/*.png` |

## Full reproduction from the raw data

```bash
N_WORKERS=8 ./reproduce.sh
```

The pipeline runs the cohort construction, case selection, population-average prescription, 3,000 participant optimizations with all baselines and scenarios, ablations on a seeded 500-participant subset, sensitivity checks, consolidation, tables, and figures. The run is deterministic. All random numbers come from NumPy `default_rng` generators with documented seeds (master seed 42; participant seeds 42 + 1009·i), and numerical libraries are single-threaded. With the pinned environment, the run reproduces the stored records exactly. The reported run took about 11 CPU-hours on 2.1 GHz Intel Xeon cores. `run.py` writes one JSON file per participant and skips participants that are already complete, so an interrupted run resumes where it stopped.

## Stored outputs

| File | Content |
|---|---|
| `calibration_cohort.csv` | 1,963-person NHANES calibration cohort (SEQN, age, sex, height, weight, BMI, waist, WHtR, GPAQ MET-min/week) |
| `participants.csv` | 3,000 synthetic participants: phenotype, latent traits, standardized scores, all lower-level coefficients, and bounds |
| `cohort_info.json` | cohort flow, copula correlation, standardization constants, model constants |
| `popavg.json` | population-average prescription |
| `case_ids.json`, `ablation_subset.json` | case-study participant identifiers; ablation subset (0-based row indices) |
| `main_results.jsonl.gz` | one record per participant: prescriptions (d, α, q, o), realized trajectories (a, f, u^R, u^E), outcomes for all six strategies, search statistics and convergence trace, solver and tie-break counts, robustness S1-S7 |
| `ablation_results.jsonl.gz` | four ablation variants per subset participant, evaluated under the full model |
| `tie_sensitivity.json` | objectives of all strategies with primary-only, nominal, and tight optimistic selection |
| `slack_check.json` | epigraph slack, reduced-Hessian null-space dimension, and flags at alternative tolerances for final incumbents |

## Citation

Please cite the associated article. Citation metadata are provided in `CITATION.cff`.

## License

The code is released under the MIT License (`LICENSE`). NHANES data are public-use data released by the U.S. National Center for Health Statistics.
