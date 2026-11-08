"""Per-machine random effect on the gate's cell priors ("machine quality"). Owner: Person D. Standard library only.

Evidence (analysis/onrate_links.py, train): a machine's on-cell and off-cell win rates are correlated (+.34 to +.52 within a
type), i.e. a latent per-machine quality moves all its cells together; across machines in an episode the sharing is weak
(ICC .07-.09, so adi's episode-level shift is the wrong level). The gate's prior treats a machine's cells as independent, so the
plentiful cells cannot inform the sparse ones. Model: each machine draws delta ~ N(0, sigma^2) on SHIFT_GRID; its cell prior
means move by delta on the log-odds scale (mode: both / on / off / contrast); P(win) = sum_delta post(delta | own history) *
P(win | delta). (mode, sigma) is learned per machine type by the marginal likelihood of TRAIN histories (no labels);
sigma = 0 is a candidate, so a missing effect leaves the gate unchanged.
"""
import math
from adi import gate as G

GRID = (-1.2, -.9, -.6, -.3, 0.0, .3, .6, .9, 1.2)
SIGMAS = (0.0, .15, .3, .45, .6, .8)
MODES = ("both", "on", "off", "contrast")


def _logprior(sigma):
    if sigma == 0:
        return [0.0 if d == 0 else -1e300 for d in GRID]
    l = [-d * d / (2 * sigma * sigma) for d in GRID]; z = G.logsumexp(l)
    return [x - z for x in l]


def _grid(gm, r, s, mode, step=None):
    cp = gm._cp(r)
    return G.win_probability_grid(r, s, gm.ST, gm._sp(s), [G.shift_cellprior(cp, d, mode) for d in GRID],
                                  **gm._mcfg(s, step=step, r_type=r["machine_type"]))


def fit_machine_shift(gm, queries, seq, step=4):
    """Choose (mode, sigma) per type; store on gm.machine_shift = {type: (mode, sigma, evidence gain)}."""
    chosen = {}
    for t in sorted({r["machine_type"] for r in queries}):
        Z = {m: [] for m in MODES}
        for r in queries:
            if r["machine_type"] == t:
                s = seq[r["machine_id"]]
                for m in MODES:
                    Z[m].append([z for _, z in _grid(gm, r, s, m, step)])
        def ev(m, sg):
            lp = _logprior(sg); return sum(G.logsumexp([a + b for a, b in zip(lp, zs)]) for zs in Z[m])
        base = ev(MODES[0], 0.0); best = (base, MODES[0], 0.0)
        for m in MODES:
            for sg in SIGMAS[1:]:
                e = ev(m, sg)
                if e > best[0]:
                    best = (e, m, sg)
        chosen[t] = (best[1], best[2], best[0] - base)
    gm.machine_shift = chosen
    return chosen


def win_prob(gm, r, s):
    """P(win) with the per-machine shift marginalized (falls back to the plain gate when sigma = 0 / not fitted)."""
    mode, sg, _ = getattr(gm, "machine_shift", {}).get(r["machine_type"], ("both", 0.0, 0.0))
    if sg == 0:
        return G.win_probability(r, s, gm.ST, gm._sp(s), cellprior=gm._cp(r), **gm._mcfg(s, r_type=r["machine_type"]))
    g = _grid(gm, r, s, mode); lp = _logprior(sg)
    post = [a + z for a, (_, z) in zip(lp, g)]; zz = G.logsumexp(post)
    return sum(math.exp(x - zz) * p for x, (p, _) in zip(post, g))
