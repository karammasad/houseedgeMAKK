"""Mixture-of-Betas cell priors for adi's change-point gate. Owner: Person D. Standard library only.

Why: the gate's on/off cell win rates vary a lot between machines (jackpot "on" rate: .27-.85) and the worst win-probability
misses are jackpot machines in the "on" cell, where a single Beta (adi) is a poor population prior. Empirical-Bayes practice
(e.g. ebbr's mixture-of-betas prior) replaces it with a mixture whose weights are learned across machines.

Fit (histories only, any pool of splits, no labels): pick machines the gate is sure about (low flip probability, a real
gate structure), take their MAP structure's per-cell counts, orient cells (higher rate = "on"), and run EM for mixture
weights over a fixed library of Beta components, separately per (machine type, signal) and role (on / off).
Use: attach_mixture(gm, queries, seq) swaps gm.cellpriors for MixCellPrior objects. adi.gate._cell_mixture is wrapped so
that tuple priors (adi) behave exactly as before and MixCellPrior objects use the mixture.
"""
import math
from adi import gate as G
from method1_gate.gate import logsumexp

NMAX = 400
MEANS = (.02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98)
STRENGTHS = (15.0, 60.0)
LIB = [(m * k, (1 - m) * k) for m in MEANS for k in STRENGTHS]


class Comp:
    """Beta(a, b) with lgamma lookup tables so integer-count evaluations are table reads."""
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.la = [math.lgamma(x + a) for x in range(NMAX)]
        self.lb = [math.lgamma(x + b) for x in range(NMAX)]
        self.ls = [math.lgamma(x + a + b) for x in range(2 * NMAX)]
        self.norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def loglik(self, n, k):          # log of BetaBinomial kernel: B(k+a, n-k+b) / B(a, b)
        return self.la[k] + self.lb[n - k] - self.ls[n] - self.norm


class Mixture:
    def __init__(self, weights, comps):
        self.w = [math.log(x) for x in weights]; self.c = comps

    def cell(self, n, k):
        """(log marginal likelihood, posterior mean rate) for a cell with n obs, k wins."""
        l = [w + c.loglik(n, k) for w, c in zip(self.w, self.c)]; z = logsumexp(l)
        pm = sum(math.exp(x - z) * (k + c.a) / (n + c.a + c.b) for x, c in zip(l, self.c))
        return z, pm


class MixCellPrior:
    def __init__(self, on, off):
        self.on, self.off = on, off


_orig = G._cell_mixture


def _cell_mixture(nz, kz, zc, cp):
    if not isinstance(cp, MixCellPrior):
        return _orig(nz, kz, zc, cp)
    nz = [int(x) for x in nz]; kz = [int(x) for x in kz]
    on = [cp.on.cell(nz[c], kz[c]) for c in (0, 1)]; off = [cp.off.cell(nz[c], kz[c]) for c in (0, 1)]
    lA = on[0][0] + off[1][0]; lB = off[0][0] + on[1][0]          # A: cell 0 is "on"; B: cell 1 is "on"
    m = max(lA, lB); wA = math.exp(lA - m); wB = math.exp(lB - m); s = wA + wB; ll = m + math.log(s / 2)
    if zc is None:                                                  # gate input missing: average by cell share
        n = nz[0] + nz[1]; f0 = (nz[0] + 1) / (n + 2)
        pA = f0 * on[0][1] + (1 - f0) * off[1][1]; pB = f0 * off[0][1] + (1 - f0) * on[1][1]
    else:
        pA = on[zc][1] if zc == 0 else off[zc][1]; pB = off[zc][1] if zc == 0 else on[zc][1]
    return ll, (wA * pA + wB * pB) / s


G._cell_mixture = _cell_mixture


def _em(counts, iters=200, keep=0.02, cap=8):
    """EM for mixture weights over LIB given per-machine (n, k) counts of one role."""
    comps = [Comp(a, b) for a, b in LIB]; w = [1 / len(LIB)] * len(LIB)
    L = [[c.loglik(n, k) for c in comps] for n, k in counts]
    for _ in range(iters):
        acc = [0.0] * len(w)
        for row in L:
            l = [math.log(x) + y for x, y in zip(w, row)]; z = logsumexp(l)
            for j, x in enumerate(l):
                acc[j] += math.exp(x - z)
        w = [max(a / len(L), 1e-12) for a in acc]
    idx = sorted(range(len(w)), key=lambda j: -w[j])[:cap]; idx = [j for j in idx if w[j] >= keep] or idx[:1]
    s = sum(w[j] for j in idx)
    return Mixture([w[j] / s for j in idx], [comps[j] for j in idx])


def machine_cells(gm, r, s, max_flip=0.1):
    """Per-cell (n, k) of the gate's MAP structure for a machine the gate is confident about, else None."""
    from method4_ml.features3 import gate_summary
    cut = len(s); g = gate_summary(s, r, gm, cells=False)
    if g[2] > max_flip or g[3] > 0.5:
        return None
    tb = G.machine_tables(s, cut, [], gm.ST, gm.cfg["a"], gm.cfg["b"], gm._cp(r))
    best = max(range(len(gm.ST)), key=lambda i: tb[i][0][0] + math.log(max(gm.sprior[i], 1e-12)))
    form, lags = gm.ST[best]; from common.data import wins
    w = wins(s); n = [0, 0]; k = [0, 0]
    for t in range(max(lags), cut):
        if w[t] < 0:
            continue
        x = G.gate_value(form, lags, w, t)
        if x is not None:
            n[x] += 1; k[x] += w[t]
    return n, k


def attach_mixture(gm, queries, seq, by="type_signal", min_group=30, max_flip=0.1):
    """Learn Mixture priors per (type[, signal]) from the given machines (histories only) and install them on gm."""
    groups = {}
    for r in queries:
        mc = machine_cells(gm, r, seq[r["machine_id"]], max_flip)
        if mc is None:
            continue
        n, k = mc; hi = 0 if (k[0] + .5) / (n[0] + 1) >= (k[1] + .5) / (n[1] + 1) else 1
        key_t = (r["machine_type"],); key_s = (r["machine_type"], r["signal"])
        for key in (key_t, key_s) if by == "type_signal" else (key_t,):
            d = groups.setdefault(key, ([], [])); d[0].append((n[hi], k[hi])); d[1].append((n[1 - hi], k[1 - hi]))
    pri = {}
    shared = ([c for g in groups.values() for c in g[0]], [c for g in groups.values() for c in g[1]])
    for key, (on, off) in sorted(groups.items(), key=lambda kv: len(kv[0])):
        if len(on) >= min_group:
            pri[key] = MixCellPrior(_em(on), _em(off))
    gm.cellpriors = pri; gm.cell_group = by
    gm.mix_n = {k: len(v[0]) for k, v in groups.items()}
    return gm
