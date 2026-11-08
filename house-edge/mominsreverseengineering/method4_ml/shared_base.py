"""Shared-base-rate + size-state-shift structure ("xsg") for the v8 gate (mominsreverseengineering). Learns from HISTORIES only.

What it does (all inside fit(), nothing hardcoded from Round 1):
  1. DERIVES candidate lag pairs (and each pair's size lag) from the gate's own fitted structure prior (gm.ST / gm.sprior): every pair family with
     prior mass >= min_mass; the size lag is the split lag of its "xs" structure when it has one, else the smaller lag.
  2. Fits, PER MACHINE TYPE, the population of base win rates by maximum marginal likelihood with every machine's rates integrated out on a logit
     grid (independent logit-normal base rate for the on side and the off side, signal slopes, size-state shift gamma, direction and on/off roles
     averaged 50/50). Two starting points, repeated L-BFGS-B restarts until converged; non-converged fits and parameters sitting on a bound are
     RECORDED in gm.shared_fit and never treated as established values.
  3. Adds one "xsg" structure per derived pair to gm.ST, writes the learned parameters into gm.cellpriors[(type, signal)][5:10] and relearns the
     structure prior with change-points marginalized (the same flip-aware accumulation adi uses for "xs"). v8's plain and split structures stay in the mixture,
     so if the size shift is absent in a new kit the data can put its weight on them (gamma is allowed to go to 0).
Requires numpy + scipy (already required by method 4).
"""
import math
from multiprocessing import get_context
import numpy as np
from scipy.optimize import minimize
from adi import gate as G

GRID = np.arange(-9.0, 7.0001, 0.2); NG = len(GRID)
SD = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])
LOGSIG = -np.logaddexp(0, -GRID); LOG1MSIG = -np.logaddexp(0, GRID)
SIG_LO, SIG_HI = math.log(0.2), math.log(1.5)
_S = {}


def lse(a, axis):
    m = a.max(axis=axis, keepdims=True); return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def derive_pairs(gm, min_mass=0.03, cap=6):
    """[((i, j), size_lag)] from the gate's fitted structure prior."""
    mass, klag = {}, {}
    for st, p in zip(gm.ST, gm.sprior):
        form, lags = st
        if len(lags) < 2 or form == "xsg":
            continue
        pair = tuple(lags[:2]); mass[pair] = mass.get(pair, 0.0) + p
        if form in ("xs", "xst") and len(lags) >= 3:
            klag[pair] = lags[2]
    keep = sorted((pr for pr in mass if mass[pr] >= min_mass), key=lambda pr: -mass[pr])[:cap]
    return [(pr, klag.get(pr, min(pr))) for pr in keep]


def extract(seq, end, pairs):
    """Counts K, N: (P, side, category) over ticks < end, plus the 'no gate' totals. A tick is used for a pair only if all its inputs are observed."""
    P = len(pairs); K = np.zeros((P, 2, 3)); N = np.zeros((P, 2, 3)); kt = nt = 0
    w = [-1 if v < 0 else int(v > 0) for v in seq]; t0 = max(max(pr[0][0], pr[0][1], pr[1]) for pr in pairs)
    for t in range(t0, end):
        if w[t] < 0:
            continue
        nt += 1; kt += w[t]
        for p, ((i, j), k) in enumerate(pairs):
            if w[t - i] < 0 or w[t - j] < 0 or seq[t - k] < 0:
                continue
            x = w[t - i] ^ w[t - j]; c = 0 if seq[t - k] == 0 else (1 if seq[t - k] == 1 else 2)
            N[p, x, c] += 1; K[p, x, c] += w[t]
    return K, N, kt, nt


class TypeFit:
    def __init__(self, K, N, kt, nt, sig, P):
        self.K, self.N, self.kt, self.nt, self.sig, self.P = K, N, kt, nt, np.asarray(sig), P
        self.names = ["mu_on", "mu_off", "ls_on", "ls_off", "b_on", "b_off", "gam", "mu_n", "ls_n"] + [f"pi{p}" for p in range(P)]
        self.bounds = [(-8, 6), (-8, 6), (SIG_LO, SIG_HI), (SIG_LO, SIG_HI), (-1.5, 1.5), (-1.5, 1.5), (0, 1.5), (-8, 6), (math.log(.2), math.log(2.0))] + [(-6, 6)] * P

    def side_ll(self, gam):
        out = []
        for d in (0, 1):
            th = GRID[None, None, None, None, :] + gam * SD[d][None, None, None, :, None]
            lp = -np.logaddexp(0, -th); lq = -np.logaddexp(0, th)
            out.append((self.K[..., None] * lp + (self.N - self.K)[..., None] * lq).sum(axis=3))   # (M, P, 2, G)
        return out

    def nll(self, x):
        v = dict(zip(self.names, x)); L = self.side_ll(v["gam"]); M = self.K.shape[0]; ev = np.zeros((M, self.P))
        so, sf = math.exp(v["ls_on"]), math.exp(v["ls_off"])
        for sg in (-1, 0, 1):
            sel = self.sig == sg
            if not sel.any():
                continue
            def lw(mu, s):
                z = -0.5 * ((GRID - mu) / s) ** 2; return z - lse(z, 0)
            won, woff = lw(v["mu_on"] + v["b_on"] * sg, so), lw(v["mu_off"] + v["b_off"] * sg, sf)
            comps = []
            for d in (0, 1):
                for w0, w1 in ((won, woff), (woff, won)):
                    comps.append(lse(L[d][sel][:, :, 0, :] + w0, -1) + lse(L[d][sel][:, :, 1, :] + w1, -1))
            ev[sel] = lse(np.stack(comps, 0), 0) - math.log(4)
        Ln = self.kt[:, None] * LOGSIG + (self.nt - self.kt)[:, None] * LOG1MSIG
        z = -0.5 * ((GRID - v["mu_n"]) / math.exp(v["ls_n"])) ** 2; Ln = Ln + (z - lse(z, 0))
        lpi = np.array([v[f"pi{p}"] for p in range(self.P)] + [0.0]); lpi = lpi - lse(lpi, 0)
        comp = np.concatenate([ev + lpi[:self.P], (lse(Ln, 1) + lpi[self.P])[:, None]], axis=1)
        return -float(lse(comp, 1).sum())

    def fit(self, starts, maxiter=400, rounds=3):
        best = None; runs = []
        for x0 in starts:
            x = np.array(x0, float); ok = False
            for _ in range(rounds):
                r = minimize(self.nll, x, method="L-BFGS-B", bounds=self.bounds, options=dict(maxiter=maxiter, eps=1e-4, ftol=1e-12, gtol=1e-6))
                x = r.x; ok = bool(r.success)
                if ok:
                    break
            runs.append((float(r.fun), ok, x))
            if best is None or r.fun < best[0]:
                best = (float(r.fun), ok, x)
        names = dict(zip(self.names, best[2])); at_bound = []
        for nm, (lo, hi), val in zip(self.names, self.bounds, best[2]):
            if min(abs(val - lo), abs(val - hi)) < 1e-3:
                at_bound.append(nm)
        spread = max(r[0] for r in runs) - min(r[0] for r in runs)
        return dict(params=names, loglik=-best[0], converged=best[1], at_bound=at_bound, start_loglik_spread=spread, x=best[2].tolist())


def _fit_type(args):
    t, = args; d = _S["data"][t]; P = _S["P"]
    tf = TypeFit(d["K"], d["N"], d["kt"], d["nt"], d["sig"], P)
    s1 = [1.0, -2.5, math.log(.5), math.log(.5), .3, .3, .4, 0.0, 0.0] + [0.0] * P
    s2 = [2.0, -3.5, math.log(.35), math.log(.8), .0, .0, .1, 0.0, 0.0] + [0.0] * P
    return t, tf.fit([s1, s2])


def attach_shared_base(gm, queries, seq, frac=0.45, min_mass=0.03, workers=3):
    """Fit the population, add the xsg structures, relearn the structure prior. Returns gm.shared_fit (also stored on gm)."""
    pairs = derive_pairs(gm, min_mass)
    if not pairs:
        gm.shared_fit = dict(pairs=[], note="no pair family above min_mass; structure not added"); return gm.shared_fit
    data = {}
    for r in queries:
        s = seq[r["machine_id"]]; short = sum(v < 0 for v in s) > len(s) / 2
        K, N, kt, nt = extract(s, len(s) if short else int(frac * len(s)), pairs)
        d = data.setdefault(r["machine_type"], dict(K=[], N=[], kt=[], nt=[], sig=[]))
        d["K"].append(K); d["N"].append(N); d["kt"].append(kt); d["nt"].append(nt); d["sig"].append(r["signal"])
    for d in data.values():
        d["K"], d["N"] = np.array(d["K"]), np.array(d["N"]); d["kt"], d["nt"] = np.array(d["kt"], float), np.array(d["nt"], float)
    _S["data"], _S["P"] = data, len(pairs); types = sorted(data)
    try:
        with get_context("fork").Pool(min(workers, len(types))) as pool:
            fits = dict(pool.map(_fit_type, [(t,) for t in types]))
    except (OSError, PermissionError, ValueError, NotImplementedError):
        fits = dict(_fit_type((t,)) for t in types)
    pri = dict(gm.cellpriors); per_type = {}
    for t, f in fits.items():
        v = f["params"]; per_type[t] = {k: (round(val, 4) if not k.startswith("ls") else round(math.exp(val), 4)) for k, val in v.items() if not k.startswith("pi")}
        per_type[t].update(loglik=round(f["loglik"], 1), converged=f["converged"], at_bound=f["at_bound"], start_spread=round(f["start_loglik_spread"], 3))
        for sg in (-1, 0, 1):
            base = tuple(pri[(t, sg)][:4]) + (0.0,) if (t, sg) in pri else None
            if base is None:
                continue
            pri[(t, sg)] = base + (v["mu_on"] + v["b_on"] * sg, v["mu_off"] + v["b_off"] * sg, math.exp(v["ls_on"]), math.exp(v["ls_off"]), v["gam"])
    gm.cellpriors = pri
    gm.ST = gm.ST + [("xsg", (pr[0], pr[1], k)) for pr, k in pairs]
    acc = [0.0] * (len(gm.ST) + 1)
    for r in queries:                                                  # structure prior with flips marginalized (same accumulation as adi's _add_splits)
        s = seq[r["machine_id"]]; cfg = gm._mcfg(s, step=4, r_type=r["machine_type"])
        taus, lp = G._taus(r["cutoff"], cfg["p_change"], cfg["step"], cfg["tau_min"])
        tb = G.machine_tables(s, r["cutoff"], taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r), cfg["half_life"])
        l = [G.logsumexp([ll + x for (ll, _), x in zip(rows, lp)]) for rows in tb]; z = G.logsumexp(l)
        for i, x in enumerate(l):
            acc[i] += math.exp(x - z)
    gm.sprior = gm._finish_prior([x / sum(acc) for x in acc])
    names = [f"{f}{lags}" for f, lags in gm.ST] + ["none"]
    gm.shared_fit = dict(pairs=[(list(pr), k) for pr, k in pairs], per_type=per_type,
                         structure_prior={n: round(p, 4) for n, p in zip(names, gm.sprior)})
    return gm.shared_fit
