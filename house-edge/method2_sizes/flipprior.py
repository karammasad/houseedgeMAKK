"""Learned flip-time prior for Adi's win gate (Method 2, experiment 11). Standard library only.

Finding (flip_time_probe.py, train histories): polarity flips are NOT uniform in time. Stable machines never flip;
drift and shared-change machines always flip, at a RELATIVE position tau / cutoff ~ 0.69 (sd ~ 0.06), not at an
absolute tick (corr(tau, cutoff) = 0.77). Adi's gate uses a flat prior: P(flip) = p_change spread uniformly over
ticks 40 .. cutoff-2.

This module replaces only that prior. For each machine adi's own tables give, per structure s and flip time tau,
the likelihood L(s, tau) of the win/loss history and the P(win) it implies. We re-weight with

    P(s, tau | history)  ∝  sprior_s * L(s, tau) * prior(tau),
    prior(tau in bin b of u = tau/cutoff) = pi_b / (#grid taus in b),   prior(no flip) = pi_none,

where pi (a histogram over relative position + no-flip mass) is learned by EM on the TRAIN machines' histories
(empirical Bayes; no conditions, labels or validation/test data). Separate pi for short logs (more than half the
ticks missing), since a flip before the observed block is invisible there. Smoothed with a [1/4, 1/2, 1/4] kernel.
"""
import math
from adi import gate as G
from adi.model import is_short

B = 40   # bins over relative position u = tau / cutoff in [0, 1)


def pieces(win, r, s):
    """adi's per-(structure, tau) likelihood and P(win), without any prior."""
    cfg = win._mcfg(s, r_type=r["machine_type"])
    taus, _ = G._taus(r["cutoff"], cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(s, r["cutoff"], taus, win.ST, cfg["a"], cfg["b"], win._cp(r), cfg.get("half_life"))
    sp = win._sp(s)
    return dict(taus=taus, cut=r["cutoff"], ls=[math.log(x) if x > 0 else None for x in sp],
                ll=[[ll for ll, _ in rows] for rows in tb], pr=[[pr for _, pr in rows] for rows in tb])


def _bins(taus, cut):
    idx = [min(B - 1, int(t / cut * B)) for t in taus]; cnt = [0] * B
    for b in idx:
        cnt[b] += 1
    return idx, cnt


def _logprior(P, pi, pinone):
    """log prior for each tau index (+ None last) under histogram pi."""
    idx, cnt = _bins(P["taus"], P["cut"])
    return [math.log(max(1e-300, pi[b] / cnt[b])) for b in idx] + [math.log(max(1e-300, pinone))]


def posterior(P, pi, pinone):
    """(P(win), mass per bin, mass on no-flip) under prior (pi, pinone)."""
    lp = _logprior(P, pi, pinone); idx, _ = _bins(P["taus"], P["cut"]); nt = len(P["taus"])
    terms = []
    for si, rows in enumerate(P["ll"]):
        if P["ls"][si] is None:
            continue
        for ti, ll in enumerate(rows):
            terms.append((ll + P["ls"][si] + lp[ti], P["pr"][si][ti], ti))
    z = G.logsumexp([x for x, _, _ in terms]); pw = 0.0; mass = [0.0] * B; mnone = 0.0
    for x, pr, ti in terms:
        w = math.exp(x - z); pw += w * pr
        if ti < nt:
            mass[idx[ti]] += w
        else:
            mnone += w
    return pw, mass, mnone


class FlipPrior:
    def __init__(self, iters=25, alpha=0.5, init_none=0.5):
        self.iters, self.alpha, self.init_none = iters, alpha, init_none

    def fit(self, win, q, seq):
        groups = {False: [], True: []}
        for r in q:
            s = seq[r["machine_id"]]; groups[is_short(s)].append(pieces(win, r, s))
        self.pi, self.pinone, self.trace = {}, {}, {}
        for short, Ps in groups.items():
            pi = [(1 - self.init_none) / B] * B; pn = self.init_none; tr = []
            for _ in range(self.iters):
                acc = [0.0] * B; an = 0.0; ll = 0.0
                for P in Ps:
                    _, m, mn = posterior(P, pi, pn)
                    for b in range(B):
                        acc[b] += m[b]
                    an += mn
                n = max(1, len(Ps))
                pn = (an + self.alpha) / (n + self.alpha * (B + 1))
                raw = [(x + self.alpha) / (n + self.alpha * (B + 1)) for x in acc]
                sm = [0.25 * raw[max(0, b - 1)] + 0.5 * raw[b] + 0.25 * raw[min(B - 1, b + 1)] for b in range(B)]
                z = pn + sum(sm); pi = [x / z for x in sm]; pn = pn / z; tr.append(pn)
            self.pi[short], self.pinone[short], self.trace[short] = pi, pn, tr
        return self

    def pwin(self, win, r, s):
        short = is_short(s)
        return posterior(pieces(win, r, s), self.pi[short], self.pinone[short])[0]

    def summary(self):
        out = {}
        for short in (False, True):
            pi, pn = self.pi[short], self.pinone[short]
            mean = sum((b + .5) / B * x for b, x in enumerate(pi)) / max(1e-12, sum(pi))
            sd = math.sqrt(sum(((b + .5) / B - mean) ** 2 * x for b, x in enumerate(pi)) / max(1e-12, sum(pi)))
            out["short" if short else "long"] = dict(p_no_flip=round(pn, 3), flip_pos_mean=round(mean, 3), flip_pos_sd=round(sd, 3))
        return out
