"""Density-ratio reweighting of the synthetic cohort toward the NHANES
calibration joint distribution (out-of-fold classifier probabilities)."""
import config  # noqa: F401
import json, glob, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
OUT = str(config.OUT_DIR)
cal = pd.read_csv(f"{OUT}/calibration_cohort.csv"); syn = pd.read_csv(f"{OUT}/participants.csv")
X = ["age", "female", "height", "weight", "waist", "activity", "bmi", "whtr"]
A = cal[X].copy(); B = syn[X].copy()
for d in (A, B): d["activity"] = np.log1p(d["activity"])
Z = np.vstack([A.values, B.values]); lab = np.r_[np.zeros(len(A)), np.ones(len(B))]
s = np.zeros(len(Z))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Z, lab):
    clf = RandomForestClassifier(500, min_samples_leaf=20, random_state=42, n_jobs=1).fit(Z[tr], lab[tr])
    s[te] = clf.predict_proba(Z[te])[:, 1]
ss = np.clip(s[len(A):], 0.02, 0.98)
w = (1 - ss) / ss * (len(B) / len(A)); w = w / w.mean()
ess = w.sum() ** 2 / (w ** 2).sum()
# check balance: weighted synthetic means vs calibration
bal = {c: [float(cal[c].mean()), float(np.average(syn[c], weights=w)), float(syn[c].mean())] for c in ["age", "bmi", "whtr", "weight", "activity"]}
np.save(f"{OUT}/cohort_diag/weights.npy", w)
json.dump(dict(ess=float(ess), w_max=float(w.max()), balance=bal), open(f"{OUT}/cohort_diag/reweight.json", "w"), indent=1)
print(ess, w.max(), bal)
