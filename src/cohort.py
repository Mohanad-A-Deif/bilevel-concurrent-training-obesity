"""NHANES 2013-2014 calibration cohort, Gaussian-copula synthetic cohort,
latent traits and participant-specific parameters (Supplement S2, S3, S6)."""
import json
import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_DIR, OUT_DIR
DATA = str(DATA_DIR)
OUT = str(OUT_DIR)

CONST = dict(
    T=16,
    d_ref=1.10, d_L=0.80, d_U=1.40, d_floor=0.35,
    f_ref=0.55, f_L=0.42, f_U=0.70,
    alpha_min=0.30, alpha_max=0.70, alpha_c=0.25, Delta_alpha=0.10,
    delta_minus=0.15, delta_plus=0.10, eps_d=1e-6,
    w_m=0.35, w_h=0.30, w_a=0.20, w_f=0.10, w_d=0.03, w_alpha=0.02,
)


# ------------------------------------------------------------------ NHANES
def _yn(s):
    return s.where(s.isin([1, 2]))


def _days(s):
    return s.where((s >= 1) & (s <= 7))


def _mins(s):
    return s.where((s >= 1) & (s <= 960))


def gpaq_met_min_week(q):
    """GPAQ total MET-min/week (vigorous x8, moderate/transport x4)."""
    domains = [("PAQ605", "PAQ610", "PAD615", 8.0),
               ("PAQ620", "PAQ625", "PAD630", 4.0),
               ("PAQ635", "PAQ640", "PAD645", 4.0),
               ("PAQ650", "PAQ655", "PAD660", 8.0),
               ("PAQ665", "PAQ670", "PAD675", 4.0)]
    total = pd.Series(0.0, index=q.index)
    valid = pd.Series(True, index=q.index)
    for yn, dd, mm, met in domains:
        y = _yn(q[yn])
        d = _days(q[dd])
        m = _mins(q[mm])
        val = np.where(y == 2, 0.0, np.where(y == 1, d * m * met, np.nan))
        val = pd.Series(val, index=q.index)
        valid &= val.notna()
        total += val.fillna(0)
    return total.where(valid)


QCOLS = ["SEQN", "PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630", "PAQ635", "PAQ640",
         "PAD645", "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675"]


def _read_components():
    """Read NHANES 2013-2014 components. Uses the official CDC transport files
    (DEMO_H.XPT, BMX_H.XPT, PAQ_H.XPT) when present in DATA_DIR; otherwise the
    CSV exports demographic.csv, examination.csv and questionnaire.csv."""
    import os
    xpt = {k: os.path.join(DATA, f"{k}.XPT") for k in ("DEMO_H", "BMX_H", "PAQ_H")}
    if all(os.path.exists(p) for p in xpt.values()):
        demo = pd.read_sas(xpt["DEMO_H"], format="xport")[["SEQN", "RIAGENDR", "RIDAGEYR", "SDDSRVYR"]]
        exam = pd.read_sas(xpt["BMX_H"], format="xport")[["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]]
        ques = pd.read_sas(xpt["PAQ_H"], format="xport")[QCOLS]
    else:
        demo = pd.read_csv(f"{DATA}/demographic.csv", usecols=["SEQN", "RIAGENDR", "RIDAGEYR", "SDDSRVYR"])
        exam = pd.read_csv(f"{DATA}/examination.csv", usecols=["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"])
        ques = pd.read_csv(f"{DATA}/questionnaire.csv", usecols=QCOLS)
    return demo, exam, ques


EXPECTED_FLOW = {"screened": 10175, "age>=20": 5769, "BMI>=30": 2083, "complete_core": 1963}


def build_calibration():
    demo, exam, ques = _read_components()
    df = demo.merge(exam, on="SEQN", how="left").merge(ques, on="SEQN", how="left")
    n_screened = len(df)
    df["activity"] = gpaq_met_min_week(df)
    df = df.rename(columns={"RIDAGEYR": "age", "BMXHT": "height", "BMXWT": "weight",
                            "BMXBMI": "bmi", "BMXWAIST": "waist"})
    df["female"] = (df["RIAGENDR"] == 2).astype(float)
    df["whtr"] = df["waist"] / df["height"]
    flow = {"screened": n_screened}
    df = df[df["age"] >= 20]; flow["age>=20"] = len(df)
    df = df[df["bmi"] >= 30]; flow["BMI>=30"] = len(df)
    core = ["age", "female", "height", "weight", "bmi", "waist", "whtr", "activity"]
    df = df.dropna(subset=core); flow["complete_core"] = len(df)
    if flow != EXPECTED_FLOW:
        import warnings
        warnings.warn(f"Cohort flow {flow} differs from the published flow {EXPECTED_FLOW}.")
    return df[["SEQN"] + core].reset_index(drop=True), flow


# ------------------------------------------------------------------ copula
PRIMARY_CONT = ["age", "height", "weight", "waist", "activity"]


class KDEMarginal:
    def __init__(self, x):
        x = np.asarray(x, float)
        kde = stats.gaussian_kde(x)
        kde.set_bandwidth(kde.scotts_factor() * 0.80)
        lo, hi = np.percentile(x, [0.25, 99.75])
        self.grid = np.linspace(lo, hi, 4096)
        pdf = kde(self.grid)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(self.grid))])
        cdf /= cdf[-1]
        cdf = np.maximum.accumulate(cdf + np.arange(cdf.size) * 1e-15)
        self.cdf = cdf / cdf[-1]

    def ppf(self, u):
        return np.interp(u, self.cdf, self.grid)


class EmpiricalLog1pMarginal:
    """Zero-inflated activity: interpolated empirical quantile function on log(1+x)."""
    def __init__(self, x):
        self.v = np.sort(np.log1p(np.asarray(x, float)))
        self.p = (np.arange(1, self.v.size + 1) - 0.5) / self.v.size

    def ppf(self, u):
        return np.expm1(np.interp(u, self.p, self.v))


def normal_scores(x):
    r = stats.rankdata(x)
    return stats.norm.ppf(r / (len(x) + 1))


def build_synthetic(cal, n_target=3000, seed=42):
    rng = np.random.default_rng(seed)
    cols = PRIMARY_CONT + ["female"]
    Z = np.column_stack([normal_scores(cal[c].values) for c in cols])
    S = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-8)
    S = V @ np.diag(w) @ V.T
    dsq = np.sqrt(np.diag(S))
    S = S / np.outer(dsq, dsq)
    margs = {c: KDEMarginal(cal[c].values) for c in PRIMARY_CONT if c != "activity"}
    margs["activity"] = EmpiricalLog1pMarginal(cal["activity"].values)
    p_female = cal["female"].mean()
    c_thr = stats.norm.ppf(1 - p_female)
    env = {c: np.percentile(cal[c].values, [0.5, 99.5]) for c in PRIMARY_CONT}
    accepted, generated, batch = [], 0, 6000
    while sum(len(a) for a in accepted) < n_target:
        G = rng.multivariate_normal(np.zeros(len(cols)), S, size=batch)
        generated += batch
        U = stats.norm.cdf(G)
        rec = {c: margs[c].ppf(U[:, j]) for j, c in enumerate(PRIMARY_CONT)}
        rec["female"] = (G[:, -1] > c_thr).astype(float)
        d = pd.DataFrame(rec)
        d["bmi"] = d["weight"] / (d["height"] / 100) ** 2
        d["whtr"] = d["waist"] / d["height"]
        ok = (d["age"] >= 20) & (d["bmi"] >= 30) & np.isfinite(d["bmi"]) & np.isfinite(d["whtr"])
        ok &= (d["whtr"] >= 0.35) & (d["whtr"] <= 1.20) & (d["activity"] >= 0)
        for c in PRIMARY_CONT:
            ok &= (d[c] >= env[c][0]) & (d[c] <= env[c][1])
        accepted.append(d[ok])
        batch = 1000
    syn = pd.concat(accepted, ignore_index=True).iloc[:n_target].reset_index(drop=True)
    info = {"generated": generated, "accepted_total": int(sum(len(a) for a in accepted)),
            "p_female": float(p_female), "corr": S.tolist(), "cols": cols}
    return syn, info


# ------------------------------------------------------------------ traits
A_ETA = np.array([2.2000, -0.1000, 0.0, -0.1800, -0.1200, 0.3000])
A_KAP = np.array([-1.7148, 0.1200, 0.0300, 0.1800, 0.1500, -0.2200])
A_RHO = np.array([-1.3863, -0.1500, 0.0, -0.1200, -0.1000, 0.2000])
A_SIG = np.array([-0.2877, -0.1200, 0.0, -0.1500, -0.1000, 0.2500])


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_participants(cal, syn):
    feats = {"age": cal["age"], "female": cal["female"], "bmi": cal["bmi"], "whtr": cal["whtr"],
             "logact": np.log1p(cal["activity"])}
    scaler = {k: (float(v.mean()), float(max(v.std(ddof=1), 1e-12))) for k, v in feats.items()}
    sv = {"age": syn["age"], "female": syn["female"], "bmi": syn["bmi"], "whtr": syn["whtr"],
          "logact": np.log1p(syn["activity"])}
    zs = {k: clip((sv[k].values - scaler[k][0]) / scaler[k][1], -3, 3) for k in scaler}
    N = len(syn)
    psi = np.column_stack([np.ones(N), zs["age"], zs["female"], zs["bmi"], zs["whtr"], zs["logact"]])
    eta, kap, rho, sig = (np.empty(N) for _ in range(4))
    for i in range(N):
        r = np.random.default_rng(42 + 1009 * (i + 1))
        e = r.normal(0, [0.12, 0.18, 0.12, 0.15])
        eta[i] = logistic(psi[i] @ A_ETA + e[0])
        kap[i] = np.exp(psi[i] @ A_KAP + e[1])
        rho[i] = logistic(psi[i] @ A_RHO + e[2])
        sig[i] = np.exp(psi[i] @ A_SIG + e[3])
    eta, kap, rho, sig = clip(eta, .70, .97), clip(kap, .10, .28), clip(rho, .10, .35), clip(sig, .45, 1.15)

    def zc(x):
        return clip((x - x.mean()) / max(x.std(ddof=1), 1e-12), -3, 3)

    te, tk, tr, ts, tA = zc(eta), zc(kap), zc(rho), zc(sig), zs["logact"]
    P = pd.DataFrame({"pid": np.arange(1, N + 1)})
    P = pd.concat([P, syn.reset_index(drop=True)], axis=1)
    P["eta"], P["kappa"], P["rho"], P["sigma"] = eta, kap, rho, sig
    P["eta_t"], P["kappa_t"], P["rho_t"], P["sigma_t"], P["A_t"] = te, tk, tr, ts, tA
    P["psiR"] = clip(1.10 * (1 + .08 * tk - .06 * ts), .75, 1.45)
    P["psiE"] = clip(1.00 * (1 + .10 * tk - .05 * ts), .70, 1.40)
    P["chi"] = clip(0.90 * (1 + .20 * tk - .10 * tr), .55, 1.40)
    P["zeta"] = clip(0.30 * (1 + .15 * te + .10 * ts), .18, .45)
    P["lam_eta"] = clip(0.30 * (1 + .10 * tr), .22, .38)
    P["lam_f"] = clip(0.045 * (1 + .20 * tk - .10 * ts), .025, .070)
    P["lam_p"] = clip(0.040 * (1 + .15 * tk - .10 * te), .020, .065)
    P["kapR"], P["kapE"], P["kapD"] = .55 * kap, .65 * kap, .20 * kap
    P["betR"] = clip(0.130 * (1 + .12 * ts - .06 * tk), .090, .170)
    P["betE"] = clip(0.110 * (1 + .10 * ts + .05 * tA - .05 * tk), .075, .145)
    P["betf"] = clip(0.028 * (1 + .20 * tk - .10 * tr), .015, .045)
    P["gamR"] = clip(0.042 * (1 + .10 * ts - .05 * tk), .028, .060)
    P["gamE"] = clip(0.051 * (1 + .12 * ts + .05 * tA - .05 * tk), .035, .070)
    P["gamo"] = clip(0.006 * (1 + .10 * te + .05 * tr), .003, .010)
    P["gamf"] = clip(0.040 * (1 + .20 * tk - .10 * tr), .022, .060)
    C = CONST
    P["dmax"] = clip(C["d_ref"] * (1 + .25 * ts + .15 * tA - .20 * tk), C["d_L"], C["d_U"])
    P["dmin"] = np.maximum(C["d_floor"], .45 * P["dmax"])
    P["d0"] = clip(.70 * P["dmax"], .55, .95)
    P["fbar"] = clip(C["f_ref"] * (1 + .20 * tk - .10 * tr), C["f_L"], C["f_U"])
    P["abar"] = clip(.72 + .05 * te - .04 * tk, .62, .82)
    P["Dd"] = clip(.12 * P["dmax"], .08, .16)
    return P, scaler


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    cal, flow = build_calibration()
    syn, info = build_synthetic(cal)
    P, scaler = build_participants(cal, syn)
    cal.to_csv(f"{OUT}/calibration_cohort.csv", index=False)
    P.to_csv(f"{OUT}/participants.csv", index=False)
    json.dump({"flow": flow, "synthetic": {k: v for k, v in info.items() if k != "corr"},
               "corr": info["corr"], "scaler": scaler, "const": CONST},
              open(f"{OUT}/cohort_info.json", "w"), indent=2)
    print(flow, info["generated"], info["accepted_total"])
    print(cal.describe().T[["mean", "std", "min", "max"]].round(2))
    print(P[["age", "bmi", "whtr", "activity", "eta", "kappa", "rho", "sigma", "dmax", "dmin", "d0", "fbar", "abar"]]
          .describe().T[["mean", "std", "min", "max"]].round(3))
