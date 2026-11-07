"""Uncertainty-adaptive size -> win feedback (Method 2, experiment 10). Standard library only.

Power posterior over win structures:  log P(s | w, y) = log P(s | w) + lam * log E_s,  lam in [0, 1].
lam = 1 is full Bayesian feedback; lam = 0 is the one-way model. Full feedback is over-confident on long logs
(misspecified size likelihood over ~100 wins), so lam should follow how uncertain the WIN evidence is.

Gates:
  "short"    lam = 1 on short logs, else 0 (experiment 9 rule; adi.model.is_short)
  "entropy"  lam = normalized entropy of P(s | w) (no parameters)
  "learned"  lam = sigmoid(a + b * H + c * short), fitted on TRAIN by maximizing the Bernoulli log-likelihood of
             next-spin win outcomes, with a N(0, 2^2) prior on (a, b, c). Examples: each train query's label plus
             self-supervised examples at earlier cutoffs (cutoff - 1, cutoff - 2) built from ticks strictly below
             the example's cutoff.
Features use only the win-only posterior and the observation pattern, never the size evidence itself, so the gate
is not circular. Validation and test answers are never read.
"""
import math
from adi.model import is_short
from method2_sizes.joint import joint_parts, assemble, lag_prior_rows

GRID = [i / 10 for i in range(11)]


def entropy_norm(P):
    """Entropy of the win-only structure posterior, divided by its maximum (log of #structures with prior > 0)."""
    p = [x for x in P["post_s"] if x > 0]; n = len({si for _, _, si in P["flat"]})
    h = -sum(x * math.log(x) for x in p)
    return h / math.log(max(2, n))


def features(P, s):
    return entropy_norm(P), float(is_short(s))


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x > -50 else 0.0


def lam_of(gate, params, H, short):
    if gate == "short":
        return short
    if gate == "entropy":
        return H
    a, b, c = params
    return sigmoid(a + b * H + c * short)


def _pw_grid(sizes, rows, r, s, P):
    """P(win) at each lam on GRID (feedback weight), for fast fitting by interpolation."""
    return [1 - assemble(sizes, rows, r, s, P, True, False, lam)[0][0] for lam in GRID]


def _interp(grid_vals, lam):
    x = lam * 10; i = min(9, int(x)); f = x - i
    return grid_vals[i] * (1 - f) + grid_vals[i + 1] * f


def training_examples(model, q, seq, extra_cutoffs=(1, 2), labels=None):
    """(features, P(win) grid, outcome) for train queries (+ earlier self-supervised cutoffs)."""
    rows = lag_prior_rows(model.win, model.sizes); out = []
    for r in q:
        full = seq[r["machine_id"]]
        cuts = [(r["cutoff"], labels.get(r["query_id"]) if labels else None)]
        cuts += [(r["cutoff"] - j, full[r["cutoff"] - j]) for j in extra_cutoffs]
        for cut, y in cuts:
            if y is None or y < 0:
                continue
            s = full[:cut]; rr = dict(r, cutoff=cut)
            P = joint_parts(model.win, model.sizes, rows, rr, s)
            out.append((features(P, s), _pw_grid(model.sizes, rows, rr, s, P), int(y > 0), r["episode_id"]))
    return out


def fit_learned_gate(examples, prior_sd=2.0):
    """MAP (a, b, c) by coordinate grid search (deterministic) on Bernoulli log-likelihood + Gaussian prior."""
    def obj(th):
        a, b, c = th; tot = -sum(x * x for x in th) / (2 * prior_sd ** 2)
        for (H, sh), pg, y, *_ in examples:
            p = min(1 - 1e-9, max(1e-9, _interp(pg, sigmoid(a + b * H + c * sh))))
            tot += math.log(p if y else 1 - p)
        return tot
    th = [0.0, 0.0, 0.0]; best = obj(th)
    for step in (2.0, 1.0, .5, .25, .125):
        improved = True
        while improved:
            improved = False
            for i in range(3):
                for d in (-step, step):
                    cand = th[:]; cand[i] += d; v = obj(cand)
                    if v > best + 1e-9:
                        th, best, improved = cand, v, True
    return th, best


def episode_cv(examples, k=5, prior_sd=2.0):
    """Episode-grouped K-fold CV on train examples (all cutoffs of one episode stay in one fold, so correlated
    self-supervised examples never straddle train/held-out). Returns out-of-fold mean log-loss and Brier score
    for each gate; 'learned' is refitted inside every fold."""
    eps = sorted({e[3] for e in examples}); fold = {ep: i % k for i, ep in enumerate(eps)}
    res = {g: [0.0, 0.0, 0] for g in ("none", "short", "entropy", "learned", "full")}; thetas = []
    for f in range(k):
        tr = [e for e in examples if fold[e[3]] != f]; te = [e for e in examples if fold[e[3]] == f]
        th, _ = fit_learned_gate(tr, prior_sd); thetas.append(th)
        for (H, sh), pg, y, _ in te:
            lams = {"none": 0.0, "full": 1.0, "short": sh, "entropy": H, "learned": lam_of("learned", th, H, sh)}
            for g, lam in lams.items():
                p = min(1 - 1e-9, max(1e-9, _interp(pg, lam)))
                res[g][0] -= math.log(p if y else 1 - p); res[g][1] += (p - y) ** 2; res[g][2] += 1
    return {g: (v[0] / v[2], v[1] / v[2]) for g, v in res.items()}, thetas
