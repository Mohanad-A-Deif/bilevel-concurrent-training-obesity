"""Bilevel prescription model: lower-level convex QP (OSQP), active-set
reduced-Hessian rank test, optimistic tie-break (CLARABEL),
and the random-forest surrogate-assisted upper-level search."""
import time
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import osqp
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor

from cohort import CONST

warnings.filterwarnings("ignore")
T = CONST["T"]
PARAM_KEYS = ["eta", "rho", "sigma", "psiR", "psiE", "chi", "zeta", "lam_eta", "lam_f", "lam_p",
              "kapR", "kapE", "kapD", "betR", "betE", "betf", "gamR", "gamE", "gamo", "gamf"]
BOUND_KEYS = ["dmax", "dmin", "d0", "fbar", "abar", "Dd"]
COEF_KEYS = ["psiR", "psiE", "chi", "zeta", "lam_eta", "lam_f", "lam_p", "kapR", "kapE", "kapD",
             "betR", "betE", "betf", "gamR", "gamE", "gamo", "gamf"]

# variable layout
iR = np.arange(T)
iE = T + np.arange(T)
iD = 2 * T + np.arange(T)
ia = 3 * T + np.arange(T + 1)
i_f = 4 * T + 1 + np.arange(T + 1)
im = 5 * T + 2 + np.arange(T + 1)
ih = 6 * T + 3 + np.arange(T + 1)
NV = 7 * T + 4

OSQP_SET = dict(eps_abs=1e-6, eps_rel=1e-6, max_iter=100000, polish=True, warm_start=True,
                adaptive_rho=True, scaled_termination=True, check_termination=25, verbose=False)
RANK_TOL = 1e-10
ACT_TOL = 1e-6


def weights(noreg=False):
    C = CONST
    return dict(m=C["w_m"], h=C["w_h"], a=C["w_a"], f=C["w_f"],
                d=0.0 if noreg else C["w_d"], al=0.0 if noreg else C["w_alpha"])


# ------------------------------------------------------------------ QP
def build_qp(p, x):
    d, al, q, o = x
    r, e = al * d, (1 - al) * d
    dprev = np.concatenate([[p["d0"]], d[:-1]])
    etaseq = p.get("etaseq", np.full(T, p["eta"]))
    Pd = np.zeros(NV)
    Pd[iR] = p["psiR"] * r ** 2
    Pd[iE] = p["psiE"] * e ** 2
    Pd[i_f[:T]] = 2 * p["chi"]
    qv = np.zeros(NV)
    qv[iR] = -p["psiR"] * r ** 2
    qv[iE] = -p["psiE"] * e ** 2
    qv[ia[:T]] = -p["zeta"]
    c0 = float(np.sum(0.5 * p["psiR"] * r ** 2 + 0.5 * p["psiE"] * e ** 2))
    rows, cols, vals, lo, hi = [], [], [], [], []
    k = [0]

    def row(entries, l, u):
        for c, v in entries:
            rows.append(k[0]); cols.append(c); vals.append(v)
        lo.append(l); hi.append(u); k[0] += 1

    # equalities
    row([(ia[0], 1)], p["eta"], p["eta"])
    row([(i_f[0], 1)], 0, 0)
    row([(im[0], 1)], 0, 0)
    row([(ih[0], 1)], p["sigma"], p["sigma"])
    seq = p["gamo"] * q * (2 * o - 1) * d
    for t in range(T):
        b = p["lam_eta"] * etaseq[t]
        row([(ia[t + 1], 1), (ia[t], -(1 - p["lam_eta"])), (i_f[t], p["lam_f"]), (iD[t], p["lam_p"])], b, b)
        row([(i_f[t + 1], 1), (i_f[t], -(1 - p["rho"])), (iR[t], -p["kapR"] * r[t]),
             (iE[t], -p["kapE"] * e[t]), (iD[t], -p["kapD"])], 0, 0)
        row([(im[t + 1], 1), (im[t], -1), (iR[t], -p["betR"] * r[t]), (iE[t], -p["betE"] * e[t]),
             (i_f[t], p["betf"])], 0, 0)
        row([(ih[t + 1], 1), (ih[t], -1), (iR[t], -p["gamR"] * r[t]),
             (iE[t], -p["gamE"] * e[t]), (i_f[t], p["gamf"]), (ia[t], -seq[t])], 0, 0)
    n_eq = k[0]
    # inequalities
    for t in range(T):
        row([(iR[t], 1), (ia[t], -1)], -np.inf, 0)
        row([(iE[t], 1), (ia[t], -1)], -np.inf, 0)
        row([(iR[t], 1)], 0, np.inf)
        row([(iE[t], 1)], 0, np.inf)
        row([(iD[t], 1)], abs(d[t] - dprev[t]), np.inf)
    for t in range(T + 1):
        row([(ia[t], 1)], 0, 1)
        row([(i_f[t], 1)], 0, p["fbar"])
    A = sp.csc_matrix((vals, (rows, cols)), shape=(k[0], NV))
    return dict(P=sp.diags(Pd).tocsc(), Pd=Pd, q=qv, c0=c0, A=A, l=np.array(lo), u=np.array(hi), n_eq=n_eq)


def j_linear(w):
    c = np.zeros(NV)
    c[im[T]] += w["m"]
    c[ih[T]] += w["h"]
    c[ia[:T]] += w["a"] / T
    c[i_f[:T]] -= w["f"] / T
    return c


def reg_term(x, p, w):
    d, al, _, _ = x
    return w["d"] * np.sum(np.diff(d) ** 2) + w["al"] * np.sum(np.diff(al) ** 2)


def reduced_hessian_deficient(Q, y):
    A = Q["A"].toarray()
    Ay = A @ y
    act = np.zeros(A.shape[0], bool)
    act[:Q["n_eq"]] = True
    ineq = np.arange(Q["n_eq"], A.shape[0])
    act[ineq] = (np.abs(Ay[ineq] - Q["l"][ineq]) <= ACT_TOL) | (np.abs(Ay[ineq] - Q["u"][ineq]) <= ACT_TOL)
    Z = sla.null_space(A[act])
    if Z.shape[1] == 0:
        return False, 0
    H = Z.T @ np.diag(Q["Pd"]) @ Z
    rk = np.linalg.matrix_rank(H, tol=RANK_TOL)
    return rk < H.shape[0], H.shape[0] - rk


import clarabel


TIE_REL, TIE_ABS = 1e-6, 1e-7


def tie_break(Q, phi_star, w):
    """Optimistic secondary problem solved directly with CLARABEL:
    min -J(y) + 1e-10||y||^2  s.t. original constraints, Phi(y) <= Phi* + tau."""
    tau = max(TIE_ABS, TIE_REL * abs(phi_star))
    Jc = j_linear(w)
    A = Q["A"].tocsr(); neq = Q["n_eq"]
    l, u = Q["l"], Q["u"]
    blocks, rhs, cones = [], [], []
    blocks.append(A[:neq]); rhs.append(l[:neq]); cones.append(clarabel.ZeroConeT(neq))
    Ain = A[neq:]; li, ui = l[neq:], u[neq:]
    fu, fl = np.isfinite(ui), np.isfinite(li)
    blocks.append(Ain[fu]); rhs.append(ui[fu])
    blocks.append(-Ain[fl]); rhs.append(-li[fl])
    cones.append(clarabel.NonnegativeConeT(int(fu.sum() + fl.sum())))
    # sum (L y)^2 <= s,  s = phi*+tau-c0-q'y ;  ||(2Ly, s-1)|| <= s+1
    nz = np.flatnonzero(Q["Pd"] > 0)
    L = np.sqrt(Q["Pd"][nz] / 2.0)
    b0 = phi_star + tau - Q["c0"]
    G = sp.lil_matrix((len(nz) + 2, NV))
    h = np.zeros(len(nz) + 2)
    # t0 = s+1 = b0+1 - q'y  ->  row: q'y, h=b0+1  (cone requires h - Gy)
    G[0, :] = Q["q"]; h[0] = b0 + 1
    G[1, :] = Q["q"]; h[1] = b0 - 1
    for k, j in enumerate(nz):
        G[2 + k, j] = -2 * L[k]
    blocks.append(G.tocsr()); rhs.append(h); cones.append(clarabel.SecondOrderConeT(len(nz) + 2))
    Afull = sp.vstack(blocks).tocsc(); b = np.concatenate(rhs)
    Pq = sp.csc_matrix(2e-10 * sp.eye(NV))
    st = clarabel.DefaultSettings()
    st.verbose = False; st.max_iter = 200
    st.tol_gap_abs = 1e-8; st.tol_gap_rel = 1e-8; st.tol_feas = 1e-8
    try:
        sol = clarabel.DefaultSolver(Pq, -Jc, Afull, b, cones, st).solve()
        if str(sol.status) == "Solved":
            return True, np.asarray(sol.x)
    except Exception:
        pass
    return False, None


class Stats:
    def __init__(self):
        self.n = 0; self.ok = 0; self.inacc = 0; self.fail = 0
        self.flag = 0; self.tie_ok = 0; self.tie_fb = 0; self.dJ = []
        self.qp_time = []

    def as_dict(self):
        return dict(n=self.n, optimal=self.ok, optimal_inaccurate=self.inacc, failed=self.fail,
                    flagged=self.flag, tie_ok=self.tie_ok, tie_fallback=self.tie_fb)


def solve_lower(p, x, w, stats=None, tiebreak=True):
    Q = build_qp(p, x)
    t0 = time.perf_counter()
    m = osqp.OSQP()
    m.setup(P=Q["P"], q=Q["q"], A=Q["A"], l=Q["l"], u=Q["u"], **OSQP_SET)
    res = m.solve()
    dt = time.perf_counter() - t0
    st = res.info.status
    resid = max(res.info.pri_res, res.info.dua_res)
    good = st in ("solved", "solved inaccurate") and resid <= 1e-5 and res.x is not None
    out = dict(ok=good, status=st, qp_time=dt, flagged=False, tie="none", dJ=0.0)
    if stats is not None:
        stats.n += 1; stats.qp_time.append(dt)
        if not good:
            stats.fail += 1
        elif st == "solved":
            stats.ok += 1
        else:
            stats.inacc += 1
    if not good:
        return out
    y = np.asarray(res.x)
    phi = 0.5 * Q["Pd"] @ y ** 2 + Q["q"] @ y + Q["c0"]
    out["phi"] = phi
    if tiebreak:
        flag, nul = reduced_hessian_deficient(Q, y)
        out["flagged"] = flag
        if flag:
            Jc = j_linear(w)
            ok2, y2 = tie_break(Q, phi, w)
            if stats is not None:
                stats.flag += 1
            if ok2:
                out["dJ"] = float(Jc @ y2 - Jc @ y)
                y = y2
                out["tie"] = "ok"
                if stats is not None:
                    stats.tie_ok += 1; stats.dJ.append(out["dJ"])
            else:
                out["tie"] = "fallback"
                if stats is not None:
                    stats.tie_fb += 1
    out["y"] = y
    return out


def outcomes(p, x, y, w):
    J = j_linear(w) @ y - reg_term(x, p, w)
    a, f = y[ia], y[i_f]
    return dict(J=float(J), m=float(y[im[T]]), h=float(y[ih[T]]), mean_a=float(a[:T].mean()),
                mean_f=float(f[:T].mean()), peak_f=float(f[:T + 1].max()), final_a=float(a[T - 1]),
                feasible=bool(a[:T].mean() >= p["abar"] - 1e-9))


def evaluate(p, x, w, stats=None, tiebreak=True):
    """Exact evaluation: lower-level QP realization + upper-level acceptability."""
    sol = solve_lower(p, x, w, stats, tiebreak)
    if not sol["ok"]:
        return dict(ok=False, feasible=False, J=np.nan, sol=sol)
    o = outcomes(p, x, sol["y"], w)
    o.update(ok=True, sol=sol)
    return o


# ------------------------------------------------------------------ upper level
C = CONST


def repair(p, x):
    d, al, q, o = (np.array(v, float) for v in x)
    q = np.round(np.clip(q, 0, 1)); o = np.round(np.clip(o, 0, 1)) * q
    lo_a = np.maximum(C["alpha_min"], C["alpha_c"] * q)
    hi_a = np.minimum(C["alpha_max"], 1 - C["alpha_c"] * q)
    al = np.clip(al, lo_a, hi_a)
    for t in range(1, T):
        al[t] = np.clip(al[t], max(lo_a[t], al[t - 1] - C["Delta_alpha"]), min(hi_a[t], al[t - 1] + C["Delta_alpha"]))
    prev, eps = p["d0"], C["eps_d"]
    for t in range(T):
        lo = max(p["dmin"], prev - p["Dd"], (1 - C["delta_minus"]) * (prev + eps) - eps)
        hi = min(p["dmax"], prev + p["Dd"], (1 + C["delta_plus"]) * (prev + eps) - eps)
        d[t] = np.clip(d[t], lo, hi)
        prev = d[t]
    return (d, al, q.astype(int), o.astype(int))


def encode(p, x):
    d, al, q, o = x
    return np.concatenate([(d - p["dmin"]) / (p["dmax"] - p["dmin"]),
                           (al - C["alpha_min"]) / (C["alpha_max"] - C["alpha_min"]), q, o])


def decode(p, z):
    d = p["dmin"] + z[:T] * (p["dmax"] - p["dmin"])
    al = C["alpha_min"] + z[T:2 * T] * (C["alpha_max"] - C["alpha_min"])
    return (d, al, z[2 * T:3 * T], z[3 * T:])


PATTERNS = np.array([[0, 0], [1, 0], [1, 1]])


def balanced_binaries(n, rng):
    qo = np.zeros((n, T, 2), int)
    for t in range(T):
        idx = np.resize(np.arange(3), n)
        rng.shuffle(idx)
        qo[:, t] = PATTERNS[idx]
    return qo


def lhs(n, seed, optimize=False):
    s = qmc.LatinHypercube(d=2 * T, scramble=True, strength=1,
                           optimization="random-cd" if optimize else None, seed=seed)
    return s.random(n)


def is_dup(z, Zarch):
    if len(Zarch) == 0:
        return False
    Za = np.asarray(Zarch)
    same_bin = np.all(Za[:, 2 * T:] == z[2 * T:], axis=1)
    if not same_bin.any():
        return False
    return bool(np.any(np.max(np.abs(Za[same_bin, :2 * T] - z[:2 * T]), axis=1) <= 1e-6))


class Search:
    def __init__(self, p, w, seed, n_init=40, n_max=250, w_stag=25, n_restart=10, tau_imp=1e-4,
                 refit_every=1, binaries_fixed=None, eval_p=None, seeds=None, struct_moves=False):
        self.p, self.w, self.seed = p, w, seed
        self.n_init, self.n_max, self.w_stag, self.n_restart = n_init, n_max, w_stag, n_restart
        self.tau_imp, self.refit_every = tau_imp, refit_every
        self.fixed = binaries_fixed  # (q,o) arrays if sequence variables are fixed
        self.rng = np.random.default_rng(seed)
        self.stats = Stats()
        self.Z, self.Jv, self.F = [], [], []
        self.best = None; self.bestJ = -np.inf; self.best_eval = 0
        self.trace = []
        self.rf = None
        self.seeds = seeds or []
        self.struct_moves = struct_moves

    def _fix(self, x):
        if self.fixed is not None:
            x = (x[0], x[1], self.fixed[0].copy(), self.fixed[1].copy())
        return repair(self.p, x)

    def _from_cont(self, U, qo):
        xs = []
        for j in range(U.shape[0]):
            z = np.concatenate([U[j], qo[j, :, 0], qo[j, :, 1]])
            xs.append(self._fix(decode(self.p, z)))
        return xs

    def _evaluate(self, x):
        o = evaluate(self.p, x, self.w, self.stats)
        z = encode(self.p, x)
        self.Z.append(z); self.Jv.append(o["J"] if o["ok"] else np.nan); self.F.append(o["ok"] and o["feasible"])
        k = len(self.Z)
        improved = False
        if self.F[-1] and o["J"] > self.bestJ:
            improved = o["J"] > self.bestJ + self.tau_imp
            self.bestJ, self.best, self.best_eval, self.best_out = o["J"], x, k, o
        self.trace.append(self.bestJ if np.isfinite(self.bestJ) else np.nan)
        return improved

    def _feasible_archive(self):
        F = np.array(self.F)
        return np.asarray(self.Z)[F], np.asarray(self.Jv)[F]

    def _exploit(self, n):
        Zf, Jf = self._feasible_archive()
        if len(Zf) == 0:
            return []
        K = min(10, int(np.ceil(0.15 * len(Zf))))
        order = np.argsort(-Jf)[:K]
        wts = np.arange(K, 0, -1, dtype=float); wts /= wts.sum()
        out = []
        for _ in range(n):
            z = Zf[order[self.rng.choice(K, p=wts)]].copy()
            x = decode(self.p, z)
            rngd = self.p["dmax"] - self.p["dmin"]
            sd, sa = 0.05 * rngd, 0.03
            d = x[0] + np.clip(self.rng.normal(0, sd, T), -2 * sd, 2 * sd)
            al = x[1] + np.clip(self.rng.normal(0, sa, T), -2 * sa, 2 * sa)
            qb, ob = x[2].copy(), x[3].copy()
            flip = self.rng.random(T) < 0.05; qb[flip] = 1 - qb[flip]
            flip = self.rng.random(T) < 0.05; ob[flip] = 1 - ob[flip]
            out.append(self._fix((d, al, qb, ob)))
        return out

    def _explore(self, n, seed, optimize=False):
        U = lhs(n, seed, optimize)
        return self._from_cont(U, balanced_binaries(n, self.rng))

    def _mutate(self, n, p_flip=0.075):
        Zf, Jf = self._feasible_archive()
        if len(Zf) == 0:
            return []
        elite = Zf[np.argsort(-Jf)[:max(1, min(10, len(Zf)))]]
        out = []
        rngd = self.p["dmax"] - self.p["dmin"]
        for j in range(n):
            base = encode(self.p, self.best) if (j % 2 == 0 and self.best is not None) else elite[self.rng.integers(len(elite))]
            d, al, qb, ob = (np.array(v, float) for v in decode(self.p, base))
            frac = self.rng.uniform(0.25, 0.50)
            sel = self.rng.random(2 * T) < frac
            dd = np.clip(self.rng.normal(0, 0.08 * rngd, T), -0.20 * rngd, 0.20 * rngd)
            da = np.clip(self.rng.normal(0, 0.05, T), -0.12, 0.12)
            d = d + dd * sel[:T]; al = al + da * sel[T:]
            flip = self.rng.random(T) < p_flip; qb[flip] = 1 - qb[flip]
            flip = self.rng.random(T) < p_flip; ob[flip] = 1 - ob[flip]
            out.append(self._fix((d, al, qb, ob)))
        return out

    def _structured(self):
        """Structured neighbourhood of the incumbent: uniform (q,o) patterns,
        single-week pattern switches, and uniform dose shifts."""
        if self.best is None:
            return []
        d, al, q, o = (np.array(v, float) for v in self.best)
        out = []
        if self.fixed is None:
            for pq, po in PATTERNS:
                out.append(self._fix((d, al, np.full(T, pq), np.full(T, po))))
            for _ in range(4):
                t = self.rng.integers(T); pq, po = PATTERNS[self.rng.integers(3)]
                q2, o2 = q.copy(), o.copy(); q2[t], o2[t] = pq, po
                out.append(self._fix((d, al, q2, o2)))
        rngd = self.p["dmax"] - self.p["dmin"]
        for sh in (-0.05, -0.02, 0.02, 0.05):
            out.append(self._fix((d + sh * rngd, al, q, o)))
        return out

    def _filter(self, xs):
        keep, Zs = [], list(self.Z)
        for x in xs:
            z = encode(self.p, x)
            if not is_dup(z, Zs):
                keep.append((x, z)); Zs.append(z)
        return keep

    def _select(self, cands, k):
        Zf, Jf = self._feasible_archive()
        if len(Zf) < 2:
            return cands[self.rng.integers(len(cands))][0]
        if self.rf is None or (k - self.last_fit) >= self.refit_every:
            ys = (Jf - Jf.mean()) / max(Jf.std(ddof=1), 1e-12)
            self.rf = RandomForestRegressor(n_estimators=128, max_depth=12, min_samples_split=4,
                                            min_samples_leaf=2, max_features=0.50, bootstrap=True,
                                            max_samples=0.85, criterion="squared_error",
                                            random_state=42, n_jobs=1).fit(Zf, ys)
            self.last_fit = k
        Zc = np.array([z for _, z in cands])
        Z32 = np.ascontiguousarray(Zc, dtype=np.float32)
        P = np.stack([e.tree_.predict(Z32).reshape(len(Z32), -1)[:, 0] for e in self.rf.estimators_])
        A = P.mean(0) + 1.25 * P.std(0, ddof=1)
        best = np.flatnonzero(A >= A.max() - 1e-12)
        if len(best) > 1:
            best = sorted(best, key=lambda j: tuple(Zc[j]))
        return cands[best[0]][0]

    def run(self):
        t0 = time.perf_counter()
        self.last_fit = -10 ** 9
        seeds = [self._fix(x) for x in self.seeds]
        init = seeds + self._explore(self.n_init - len(seeds), self.seed, optimize=True)
        for x, _ in self._filter(init):
            self._evaluate(x)
        k, s, r = len(self.Z), 0, 0
        stop = None
        while True:
            if k >= self.n_max:
                stop = "budget"; break
            if s >= self.w_stag:
                stop = "stagnation"; break
            if r >= self.n_restart:
                stop = "restart_limit"; break
            pool = self._exploit(26) + self._explore(26, self.seed + 7919 * (r + 1) + k) + self._mutate(12)
            if self.struct_moves:
                pool += self._structured()
            cands = self._filter(pool)
            if not cands:
                r += 1
                rs = self._explore(52, self.seed + 7919 * r) + self._mutate(12, 0.10)
                cands = self._filter(rs)
                if not cands:
                    continue
            x = self._select(cands, k)
            improved = self._evaluate(x)
            k += 1
            s = 0 if improved else s + 1
        self.runtime = time.perf_counter() - t0
        self.stop, self.k, self.restarts = stop, k, r
        return self


def refine_cma(S, budget, sigma0=0.10, seed=None):
    """CMA-ES refinement of the incumbent of a completed surrogate-assisted search S.
    Continuous variables are searched in the scaled box [0,1]^{2T}; each weekly
    (q_t, o_t) pattern is encoded by one coordinate in [0,1] split into three equal
    intervals. Every candidate is repaired and evaluated exactly, and the search
    statistics, incumbent and trace of S are updated in place."""
    import cma
    if S.best is None or S.k >= budget:
        S.k2 = 0
        return S
    t0 = time.perf_counter()
    e = encode(S.p, S.best)
    pat = np.where(e[2 * T:3 * T] == 0, 0, np.where(e[3 * T:] == 0, 1, 2))
    z0 = np.concatenate([np.clip(e[:2 * T], 0, 1), (pat + 0.5) / 3.0])
    opts = {"bounds": [0, 1], "seed": int((S.seed if seed is None else seed) % (2 ** 31 - 1)) + 1,
            "verbose": -9, "tolfun": 1e-9, "tolx": 1e-7}
    es = cma.CMAEvolutionStrategy(z0, sigma0, opts)
    k2 = 0
    while S.k < budget and not es.stop():
        Zs = es.ask()
        F = []
        for z in Zs:
            if S.k >= budget:
                break
            z = np.clip(np.asarray(z, float), 0, 1)
            pp = np.minimum((z[2 * T:] * 3).astype(int), 2)
            qo = PATTERNS[pp]
            x = S._fix(decode(S.p, np.concatenate([z[:2 * T], qo[:, 0], qo[:, 1]])))
            o = evaluate(S.p, x, S.w, S.stats)
            S.k += 1; k2 += 1
            if not o["ok"]:
                fit = -10.0
            elif o["feasible"]:
                fit = o["J"]
                if fit > S.bestJ:
                    S.bestJ, S.best, S.best_eval, S.best_out = fit, x, S.k, o
            else:
                fit = o["J"] - 1.0 - 10.0 * (S.p["abar"] - o["mean_a"])
            S.trace.append(S.bestJ)
            F.append(-fit)
        if len(F) < len(Zs):
            break
        es.tell(Zs, F)
    S.k2 = k2
    S.stop2 = "budget" if S.k >= budget else "cma_stop"
    S.runtime += time.perf_counter() - t0
    return S


def search_hybrid(p, w, seed, budget=2500, **kw):
    """Two-stage upper-level search: warm-started random-forest surrogate-assisted
    stage (Algorithm 1) followed by CMA-ES refinement of its incumbent."""
    S = Search(p, w, seed, **kw).run()
    S.k1, S.stop1 = S.k, S.stop
    return refine_cma(S, budget)


def average_participant(P):
    keys = PARAM_KEYS + BOUND_KEYS
    return {k: float(P[k].mean()) for k in keys}
