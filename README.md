# Bilevel optimization of concurrent resistance–endurance training in obesity

Code, data, and outputs for the in silico study **"Personalized Concurrent Resistance–Endurance Training in Obesity: A Bilevel Framework for Dose and Sequence Optimization"** (Deif, Elhaty, Hafez, and Khishe).

The repository contains every step of the analysis. It calibrates a baseline phenotype space on NHANES 2013–2014 and generates a synthetic cohort of 3,000 adults with obesity. For each participant, it optimizes a 16-week concurrent-training prescription with a bilevel model, compares the result with five baseline strategies, and runs the ablations, robustness scenarios, and numerical sensitivity checks. It also contains the upper-level solver benchmark (GA, TPE, CMA-ES, and a KKT reformulation solved with SCIP), the synthetic-cohort fidelity diagnostics, and the latent-trait sensitivity analysis. Every table and figure of the article can be regenerated from the stored outputs in about a minute, or recomputed from the raw data.

> **Scope.** All results are in silico. NHANES is used only to calibrate baseline phenotypes; the training-response dynamics are prespecified modeling assumptions. The outputs are not clinical exercise prescriptions.

## Model in brief

- **Upper level (prescription).** For each week *t = 1..16*, the upper level chooses the total dose *d_t*, the resistance share *α_t*, same-session concurrency *q_t ∈ {0,1}*, and resistance→endurance order *o_t ∈ {0,1}*. It maximizes a weighted objective of terminal body-composition benefit, terminal fitness, mean adherence, and mean fatigue, minus smoothness penalties. Bounds, change limits, and progression limits apply.
- **Lower level (realization).** A participant-specific convex quadratic program returns the executed fractions and the adherence, fatigue, body-composition, and fitness trajectories for a given prescription. It is solved with OSQP.
- **Optimistic selection.** If the active-set reduced Hessian is rank deficient, an ε-optimistic secondary problem (solved with CLARABEL) selects the realization with the largest upper-level objective.
- **Upper-level search (two stages).** A random-forest surrogate-assisted mixed-variable heuristic is warm-started from three rule-based prescriptions (at most 250 exact evaluations, stagnation window 50). Its incumbent is then refined by CMA-ES until a total of 1000 exact evaluations per participant.

All parameter values are listed in `src/cohort.py` (`CONST` and the coefficient mappings) and `src/model.py              lower-level QP, reduced-Hessian test, optimistic tie-break, surrogate-assisted search, CMA-ES refinement
src/run.py                proposed framework, baselines, robustness scenarios S1-S7, ablations (checkpointed)
src/sensitivity.py        re-evaluation with primary-only and tighter tie bands
src/slack_check.py        epigraph slack, terminal-constraint activity, rank/active-set tolerance checks
src/bench.py              upper-level solver benchmark (GA/pymoo, TPE/Optuna, CMA-ES) on 204 participants
src/kkt.py, kkt_cases.py  single-level KKT reformulation solved with SCIP for the case-study participants
src/cohort_diag.py        synthetic-cohort fidelity diagnostics (correlations, energy distance, classifier test)
src/reweight.py           density-ratio weights of the synthetic cohort toward the calibration cohort
src/cohort_variants.py    latent-trait generation variants; sens_latent.py runs the sensitivity analysis
src/consolidate.py        packs per-participant records into outputs/*_results.jsonl.gz
src/analyze.py            aggregation functions used by the tables and figures
src/analyze_extra.py      two-stage contribution, benchmark, KKT, reweighting, and latent-trait summaries
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
| Table 10 (population benchmark) | `make_tables.py` | `outputs/tables/table10_population_benchmark.csv` |
| Table 11 (paired differences) | `make_tables.py` | `table11_paired_differences.csv` |
| Table 12 (ablations, 500-participant subset) | `make_tables.py` | `table12_ablation.csv` |
| Table 13 (optimistic-selection sensitivity) | `make_tables.py` | `table13_tie_sensitivity.csv` |
| Table 14 (computational performance, tie handling, stage contribution) | `make_tables.py` | `table14_computational.json` |
| Table 15 (upper-level solver benchmark) | `make_tables.py` | `table15_solver_benchmark.csv` |
| Table 16 (KKT reformulation bounds) | `make_tables.py` | `table16_kkt_bounds.csv` |
| Table 17 (latent-trait sensitivity) | `make_tables.py` | `table17_latent_sensitivity.csv` |
| Table 18 (S7 stress test) | `make_tables.py` | `table18_s7.csv` |
| Table 19 (robustness S0-S6) | `make_tables.py` | `table19_robustness.csv` |
| Table 20 (case studies) | `make_tables.py` | `table20_case_studies.csv` |
| Reweighted comparisons, prescription structure, (q, o) counts | `make_tables.py` | `reweighted_comparisons.csv`, `prescription_structure.json`, `sequence_counts.json` |
| Supplementary cohort agreement and constraint margins | `make_tables.py` | `tableS6_cohort_fidelity.csv`, `tableS7_constraint_margins.csv` |
| Supplementary fidelity diagnostics (S12) | `cohort_diag.py`, `reweight.py` | `outputs/cohort_diag/` |
| Figures 2-3 data | `make_tables.py` | `figures2_3_weekly_*.csv` |
| Figure 4 data | `make_tables.py` | `figure4_convergence.csv` |
| Figures 2-6 | `figures.py` | `figures/*.png` |

## Full reproduction from the raw data

```bash
N_WORKERS=8 ./reproduce.sh
```

The pipeline runs the cohort construction, case selection, population-average prescription, 3,000 participant optimizations with all baselines and scenarios, ablations on a seeded 500-participant subset, sensitivity checks, cohort diagnostics, the solver benchmark, the KKT runs, the latent-trait sensitivity analysis, consolidation, tables, and figures. The run is deterministic. All random numbers come from NumPy `default_rng` generators with documented seeds (master seed 42; participant seeds 42 + 1009·i), and numerical libraries are single-threaded. With the pinned environment, the run reproduces the stored records exactly. The main 3,000-participant run took about 20 CPU-hours (11.7 h of search time) on 2.1 GHz Intel Xeon cores; the ablations, benchmark, KKT runs (4 x 1800 s), and sensitivity analyses took about 15 further CPU-hours. `run.py` writes one JSON file per participant and skips participants that are already complete, so an interrupted run resumes where it stopped.

## Stored outputs

| File | Content |
|---|---|
| `calibration_cohort.csv` | 1,963-person NHANES calibration cohort (SEQN, age, sex, height, weight, BMI, waist, WHtR, GPAQ MET-min/week) |
| `participants.csv` | 3,000 synthetic participants: phenotype, latent traits, standardized scores, all lower-level coefficients, and bounds |
| `cohort_info.json` | cohort flow, copula correlation, standardization constants, model constants |
| `popavg.json` | population-average prescription |
| `case_ids.json`, `ablation_subset.json` | case-study participant identifiers; ablation subset (0-based row indices) |
| `main_results.jsonl.gz` | one record per participant: prescriptions (d, α, q, o), realized trajectories (a, f, u^R, u^E), outcomes for all six strategies, search statistics (both stages) and convergence trace, solver and tie-break counts, robustness S1-S7 |
| `ablation_results.jsonl.gz` | four ablation variants per subset participant, evaluated under the full model |
| `bench_results.jsonl.gz` | solver-benchmark records (field `tag` = solver and budget; `kkt` = SCIP runs with dual bounds) |
| `sens_latent_results.jsonl.gz` | latent-trait sensitivity records (field `variant`) |
| `tie_sensitivity.json` | objectives of all strategies with primary-only, nominal, and tight optimistic selection |
| `slack_check.json` | epigraph slack, reduced-Hessian null-space dimension, and flags at alternative tolerances for final incumbents |
| `cohort_diag/` | correlation matrices, joint-distribution tests, sex-stratified comparisons, and density-ratio weights |

## Citation

Please cite the associated article. Citation metadata are provided in `CITATION.cff`.

## License

The code is released under the MIT License (`LICENSE`). NHANES data are public-use data released by the U.S. National Center for Health Statistics.
