import os
import config  # noqa: F401  (single-threaded numerical libraries)
"""Upper-level solver benchmark on the case-study participants plus a seeded
random subset. Every solver uses the same exact evaluation (lower-level QP with
optimistic selection), the same repair operator and the same budget of exact
evaluations. Usage: python bench.py <method> <start> <end> [budget]"""
import sys, json, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from model import T, Search, repair, evaluate, encode, decode, PATTERNS, lhs
from run import P, W, seed_of, rule_seeds, REFIT, W_STAG

OUT = str(config.OUT_DIR)
BDIR = f"{OUT}/bench"
os.makedirs(BDIR, exist_ok=True)
NB = 200


def bench_ids():
    cases = list(json.load(open(f"{OUT}/case_ids.json")).values())
    sub = json.load(open(f"{OUT}/ablation_subset.json"))  # 0-based rows, seeded random
    rows = {int(P.iloc[i]["pid"]): i for i in range(len(P))}
    pids = [int(P.iloc[i]["pid"]) for i in sub if int(P.iloc[i]["pid"]) not in cases][:NB]
    return [rows[c] for c in cases] + [rows[p] for p in pids]


# --------------------------------------------------------------- common objective
class Obj:
    """Wraps the exact bilevel evaluation; tracks the best realized-feasible design."""

    def __init__(self, p, budget):
        self.p, self.budget = p, budget
        self.n = 0; self.bestJ = -np.inf; self.best = None; self.best_eval = 0; self.trace = []

    def __call__(self, z):
        if self.n >= self.budget:
            return None
        z = np.clip(np.asarray(z, float), 0, 1)
        pat = np.minimum((z[2 * T:] * 3).astype(int), 2)
        qo = PATTERNS[pat]
        x = repair(self.p, decode(self.p, np.concatenate([z[:2 * T], qo[:, 0], qo[:, 1]])))
        o = evaluate(self.p, x, W)
        self.n += 1
        if not o["ok"]:
            fit = -10.0
        elif o["feasible"]:
            fit = o["J"]
            if fit > self.bestJ:
                self.bestJ, self.best, self.best_eval = fit, x, self.n
        else:
            fit = o["J"] - 1.0 - 10.0 * (self.p["abar"] - o["mean_a"])
        self.trace.append(self.bestJ if np.isfinite(self.bestJ) else np.nan)
        return fit


def z_of(p, x):
    e = encode(p, x)
    pat = np.where(e[2 * T:3 * T] == 0, 0, np.where(e[3 * T:] == 0, 1, 2))
    return np.concatenate([np.clip(e[:2 * T], 0, 1), (pat + 0.5) / 3.0])


def warm_z(p):
    return [z_of(p, x) for x in rule_seeds(p)]


def init_design(p, n, seed, warm):
    U = lhs(n, seed, optimize=True)
    rng = np.random.default_rng(seed)
    Z = np.concatenate([U, rng.random((n, T))], axis=1)
    if warm:
        W0 = np.array(warm_z(p)); Z[:len(W0)] = W0
    return Z


# --------------------------------------------------------------- solvers
def run_ga(p, seed, budget, warm):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    ob = Obj(p, budget)

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=3 * T, n_obj=1, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *a, **k):
            F = []
            for z in X:
                v = ob(z)
                F.append(1e3 if v is None else -v)
            out["F"] = np.array(F)

    pop = 40
    alg = GA(pop_size=pop, sampling=init_design(p, pop, seed, warm), crossover=SBX(prob=0.9, eta=15),
             mutation=PM(eta=20), eliminate_duplicates=True)
    minimize(Prob(), alg, ("n_evals", budget), seed=seed % (2 ** 31), verbose=False)
    return ob


def run_cma(p, seed, budget, warm):
    import cma
    ob = Obj(p, budget)
    x0 = warm_z(p)[2] if warm else np.full(3 * T, 0.5)
    es = cma.CMAEvolutionStrategy(x0, 0.2, {"bounds": [0, 1], "seed": seed % (2 ** 31), "verbose": -9,
                                            "maxfevals": budget})
    if warm:
        ob(warm_z(p)[0]); ob(warm_z(p)[1]); ob(x0)
    while ob.n < budget and not es.stop():
        Xs = es.ask()
        F = []
        for z in Xs:
            v = ob(z)
            F.append(1e3 if v is None else -v)
        if ob.n >= budget:
            break
        es.tell(Xs, F)
    return ob


def run_tpe(p, seed, budget, warm):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    ob = Obj(p, budget)
    smp = optuna.samplers.TPESampler(seed=seed % (2 ** 31), n_startup_trials=40, multivariate=True)
    st = optuna.create_study(direction="maximize", sampler=smp)
    names = [f"u{j}" for j in range(2 * T)]
    if warm:
        for z in warm_z(p):
            pr = {n: float(z[j]) for j, n in enumerate(names)}
            pr.update({f"p{t}": int(z[2 * T + t] * 3) for t in range(T)})
            st.enqueue_trial(pr)

    def f(tr):
        z = [tr.suggest_float(n, 0.0, 1.0) for n in names]
        z += [(tr.suggest_categorical(f"p{t}", [0, 1, 2]) + 0.5) / 3.0 for t in range(T)]
        return ob(np.array(z))

    st.optimize(f, n_trials=budget)
    return ob


def run_prop(p, seed, budget, warm, stag=W_STAG):
    S = Search(p, W, seed=seed, n_max=budget, refit_every=REFIT, w_stag=stag,
               seeds=rule_seeds(p) if warm else [], struct_moves=True).run()
    ob = Obj(p, budget)
    ob.n, ob.bestJ, ob.best, ob.best_eval, ob.trace = S.k, S.bestJ, S.best, S.best_eval, list(S.trace)
    return ob


def run_hybrid(p, seed, budget, stage1=250, sigma0=0.10):
    """Surrogate-assisted global stage followed by CMA-ES refinement from the incumbent."""
    import cma
    S = Search(p, W, seed=seed, n_max=stage1, refit_every=REFIT, w_stag=W_STAG,
               seeds=rule_seeds(p), struct_moves=True).run()
    ob = Obj(p, budget)
    ob.n, ob.bestJ, ob.best, ob.best_eval, ob.trace = S.k, S.bestJ, S.best, S.best_eval, list(S.trace)
    x0 = z_of(p, S.best)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {"bounds": [0, 1], "seed": seed % (2 ** 31), "verbose": -9})
    while ob.n < budget and not es.stop():
        Xs = es.ask(); F = []
        for z in Xs:
            v = ob(z); F.append(1e3 if v is None else -v)
        if ob.n >= budget:
            break
        es.tell(Xs, F)
    return ob


METHODS = {
    "prop_cold": lambda p, s, b: run_prop(p, s, b, False),
    "prop_full": lambda p, s, b: run_prop(p, s, b, True, stag=10 ** 9),
    "ga_warm": lambda p, s, b: run_ga(p, s, b, True),
    "ga_cold": lambda p, s, b: run_ga(p, s, b, False),
    "tpe_warm": lambda p, s, b: run_tpe(p, s, b, True),
    "tpe_cold": lambda p, s, b: run_tpe(p, s, b, False),
    "cma_warm": lambda p, s, b: run_cma(p, s, b, True),
    "cma_cold": lambda p, s, b: run_cma(p, s, b, False),
    "hybrid": lambda p, s, b: run_hybrid(p, s, b),
}


def job(args):
    method, i, budget = args
    pid = int(P.iloc[i]["pid"])
    tag = method if budget == 250 else f"{method}_b{budget}"
    path = f"{BDIR}/{tag}/{pid:04d}.json"
    if os.path.exists(path):
        return pid
    p = P.iloc[i].to_dict()
    t0 = time.perf_counter()
    ob = METHODS[method](p, seed_of(pid) + 104729, budget)
    rt = time.perf_counter() - t0
    rec = dict(pid=pid, method=tag, J=None if ob.best is None else float(ob.bestJ), n=ob.n,
               best_eval=ob.best_eval, runtime=rt,
               trace=[None if not np.isfinite(v) else round(float(v), 6) for v in ob.trace])
    if ob.best is not None:
        rec["d"] = np.round(ob.best[0], 5).tolist(); rec["alpha"] = np.round(ob.best[1], 5).tolist()
        rec["q"] = np.asarray(ob.best[2]).astype(int).tolist(); rec["o"] = np.asarray(ob.best[3]).astype(int).tolist()
    json.dump(rec, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return pid


if __name__ == "__main__":
    method = sys.argv[1]; a, b = int(sys.argv[2]), int(sys.argv[3])
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 250
    tag = method if budget == 250 else f"{method}_b{budget}"
    os.makedirs(f"{BDIR}/{tag}", exist_ok=True)
    idx = bench_ids()[a:b]
    t0 = time.time()
    with Pool(config.N_WORKERS) as pool:
        for n, pid in enumerate(pool.imap_unordered(job, [(method, i, budget) for i in idx], chunksize=1)):
            if n % 20 == 0:
                print(f"{tag} {n + 1}/{len(idx)} pid={pid} {time.time() - t0:.0f}s", flush=True)
    print("FINISHED", tag, time.time() - t0, flush=True)
