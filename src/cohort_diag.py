"""Synthetic-cohort fidelity diagnostics: marginal, dependence, joint and
plausibility comparisons between the NHANES calibration cohort and the
accepted synthetic cohort."""
import config  # noqa: F401
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

OUT = str(config.OUT_DIR)
D = f"{OUT}/cohort_diag"
import os
os.makedirs(D, exist_ok=True)
cal = pd.read_csv(f"{OUT}/calibration_cohort.csv")
syn = pd.read_csv(f"{OUT}/participants.csv")
V = ["age", "female", "height", "weight", "waist", "activity", "bmi", "whtr"]
cal = cal[V].copy(); syn = syn[V].copy()
for d in (cal, syn):
    d["logact"] = np.log1p(d["activity"])
X = ["age", "female", "height", "weight", "waist", "logact", "bmi", "whtr"]
res = {}

# dependence
Sc = cal[X].corr("spearman"); Ss = syn[X].corr("spearman")
Sc.to_csv(f"{D}/spearman_cal.csv"); Ss.to_csv(f"{D}/spearman_syn.csv")
Dm = (Ss - Sc).abs().values[np.triu_indices(len(X), 1)]
res["spearman_absdiff_max"] = float(Dm.max()); res["spearman_absdiff_mean"] = float(Dm.mean())
Pc = cal[X].corr(); Ps = syn[X].corr()
res["pearson_absdiff_max"] = float((Ps - Pc).abs().values[np.triu_indices(len(X), 1)].max())

# energy distance on standardized vectors with permutation test and split-half reference
mu, sd = cal[X].mean(), cal[X].std()
A = ((cal[X] - mu) / sd).values; B = ((syn[X] - mu) / sd).values


def edist(A, B, rng=None, m=1500):
    if rng is not None:
        A = A[rng.choice(len(A), min(m, len(A)), replace=False)]
        B = B[rng.choice(len(B), min(m, len(B)), replace=False)]
    from scipy.spatial.distance import cdist
    return 2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean()


rng = np.random.default_rng(42)
e_obs = edist(A, B)
Z = np.vstack([A, B]); nA = len(A)
perm = []
for _ in range(200):
    idx = rng.permutation(len(Z)); perm.append(edist(Z[idx[:nA]], Z[idx[nA:]]))
res["energy_distance"] = float(e_obs)
res["energy_perm_p"] = float((np.sum(np.array(perm) >= e_obs) + 1) / 201)
res["energy_perm_null_p95"] = float(np.percentile(perm, 95))
# per-variable-set energy distances: excluding BMI/WHtR (derived) and primary only
for name, cols in {"primary": ["age", "female", "height", "weight", "waist", "logact"]}.items():
    j = [X.index(c) for c in cols]
    res[f"energy_{name}"] = float(edist(A[:, j], B[:, j]))

# classifier two-sample test
lab = np.r_[np.zeros(len(A)), np.ones(len(B))]
aucs = []; imp = np.zeros(len(X))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Z, lab):
    clf = RandomForestClassifier(500, min_samples_leaf=5, random_state=42, n_jobs=1).fit(Z[tr], lab[tr])
    aucs.append(roc_auc_score(lab[te], clf.predict_proba(Z[te])[:, 1])); imp += clf.feature_importances_ / 5
res["c2st_auc_mean"] = float(np.mean(aucs)); res["c2st_auc_sd"] = float(np.std(aucs, ddof=1))
res["c2st_importance"] = dict(zip(X, np.round(imp, 3).tolist()))
aucs2 = []
Zp = Z[:, :6]
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Zp, lab):
    clf = RandomForestClassifier(500, min_samples_leaf=5, random_state=42, n_jobs=1).fit(Zp[tr], lab[tr])
    aucs2.append(roc_auc_score(lab[te], clf.predict_proba(Zp[te])[:, 1]))
res["c2st_auc_primary_only"] = float(np.mean(aucs2))

# sex-stratified plausibility
sexs = {}
for s, nm in [(1.0, "female"), (0.0, "male")]:
    c = cal[cal.female == s]; y = syn[syn.female == s]
    sexs[nm] = {v: dict(cal_mean=float(c[v].mean()), syn_mean=float(y[v].mean()),
                        ks=float(stats.ks_2samp(c[v], y[v]).statistic))
                for v in ["age", "height", "weight", "waist", "bmi", "whtr", "activity"]}
res["by_sex"] = sexs
# BMI classes
bins = [30, 35, 40, np.inf]
res["bmi_class_cal"] = (pd.cut(cal.bmi, bins, right=False).value_counts(normalize=True).sort_index() * 100).round(1).tolist()
res["bmi_class_syn"] = (pd.cut(syn.bmi, bins, right=False).value_counts(normalize=True).sort_index() * 100).round(1).tolist()
# plausibility ranges
res["ranges_syn"] = {v: [float(syn[v].min()), float(syn[v].max())] for v in ["age", "height", "weight", "waist", "bmi", "whtr", "activity"]}
res["ranges_cal"] = {v: [float(cal[v].min()), float(cal[v].max())] for v in ["age", "height", "weight", "waist", "bmi", "whtr", "activity"]}
# quantile comparison for BMI
res["bmi_quantiles"] = {str(qq): [float(cal.bmi.quantile(qq)), float(syn.bmi.quantile(qq))] for qq in [.1, .25, .5, .75, .9, .99]}
json.dump(res, open(f"{D}/diagnostics.json", "w"), indent=1)
print(json.dumps(res, indent=1))
