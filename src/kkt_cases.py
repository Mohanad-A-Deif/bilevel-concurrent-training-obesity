import os
import config  # noqa: F401  (single-threaded numerical libraries)
"""KKT/SCIP runs for the four case-study participants, warm-started from the
two-stage solution. Usage: python kkt_cases.py"""
import json
import numpy as np
from multiprocessing import Pool
import kkt
from run import P

OUT = str(config.OUT_DIR)
D = f"{OUT}/bench/kkt"


def job(pid):
    i = int(P.index[P.pid == pid][0])
    r = json.load(open(f"{OUT}/main/{pid:04d}.json"))["proposed"]
    x0 = (np.array(r["d"]), np.array(r["alpha"]), np.array(r["q"]), np.array(r["o"]))
    kkt.solve_from(i, x0, 1800, D)
    return pid


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    ids = list(json.load(open(f"{OUT}/case_ids.json")).values())
    with Pool(config.N_WORKERS) as pool:
        for pid in pool.imap_unordered(job, ids):
            print("kkt done", pid, flush=True)
