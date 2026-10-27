"""Latent-trait generation variants (sensitivity analysis). With default
arguments build_participants_v reproduces cohort.build_participants exactly."""
import config  # noqa: F401
import numpy as np, pandas as pd
from cohort import CONST, A_ETA, A_KAP, A_RHO, A_SIG, clip, logistic


def build_participants_v(cal, syn, slope=1.0, noise=1.0, mapping=1.0):
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
        e = noise * r.normal(0, [0.12, 0.18, 0.12, 0.15])
        eta[i] = logistic(psi[i] @ np.r_[A_ETA[0], slope * A_ETA[1:]] + e[0])
        kap[i] = np.exp(psi[i] @ np.r_[A_KAP[0], slope * A_KAP[1:]] + e[1])
        rho[i] = logistic(psi[i] @ np.r_[A_RHO[0], slope * A_RHO[1:]] + e[2])
        sig[i] = np.exp(psi[i] @ np.r_[A_SIG[0], slope * A_SIG[1:]] + e[3])
    eta, kap, rho, sig = clip(eta, .70, .97), clip(kap, .10, .28), clip(rho, .10, .35), clip(sig, .45, 1.15)

    def zc(x):
        return clip((x - x.mean()) / max(x.std(ddof=1), 1e-12), -3, 3)

    te, tk, tr, ts, tA = zc(eta), zc(kap), zc(rho), zc(sig), zs["logact"]
    te, tk, tr, ts, tA = (mapping * v for v in (te, tk, tr, ts, tA))
    P_t = (te, tk, tr, ts, tA)
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


