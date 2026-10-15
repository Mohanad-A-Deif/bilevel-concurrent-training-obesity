"""Select the four representative case-study participants from phenotype and
latent-trait criteria (independent of any optimization result)."""
import json
import numpy as np
import pandas as pd
from config import OUT_DIR


def pick(sub, score):
    target = np.percentile(score, 97.5)          # linear interpolation between order statistics
    return int(sub.pid[(score - target).abs().idxmin()])  # first minimum = smallest participant id


if __name__ == "__main__":
    P = pd.read_csv(OUT_DIR / "participants.csv")
    cases = {"A": pick(P, P.eta_t - P.kappa_t + P.rho_t)}
    sb = P[(P.rho_t >= -1) & (P.rho_t <= 0.5)]; cases["B"] = pick(sb, -sb.eta_t + sb.kappa_t)
    sc = P[(P.rho_t.abs() <= 0.5) & (P.kappa_t.abs() <= 0.5)]; cases["C"] = pick(sc, -sc.A_t)
    sd = P[P.eta_t.abs() <= 0.5]; cases["D"] = pick(sd, -(sd.rho_t + sd.sigma_t))
    json.dump(cases, open(OUT_DIR / "case_ids.json", "w"))
    print(cases, "eligible sets:", len(P), len(sb), len(sc), len(sd))
