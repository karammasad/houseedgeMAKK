"""Bidirectional Bayesian win x size inference (Method 2, experiment 9). Standard library only.

Joint posterior over adi's win-gate structure s (with flips tau averaged) and Method 2's size lag k:

    P(s, tau, k | w, y)  ∝  P(s) P(tau) P(w | s, tau)  *  pi_s(k)  *  P(y | w, k)

  P(w | s, tau)   adi's gate likelihood of the win/loss sequence (past sizes enter only as inputs for "xs")
  P(y | w, k)     Method 2's likelihood of the size outcomes (1x vs big) of the wins, given the lag-k categories
  pi_s(k)         (1 - c) * C[s][k] + c * g[k]: the learned coupling table (row of s's lag pair) + global prior

No double counting: by the chain rule every tick's win/loss outcome is in P(w|s,tau) once and every win's size
outcome is in P(y|w,k) once; past values are only conditioning inputs.

  size -> win feedback : P(win) = sum_{s,tau} P(s,tau | w, y) * pw_{s,tau},   P(s,tau|w,y) ∝ P(s,tau|w) * E_s,
                         E_s = sum_k pi_s(k) L_k   (L_k = size evidence at lag k)
  win -> size (exact)  : P(k | w, y) ∝ [sum_s P(s | w) pi_s(k)] * L_k   (adi's posterior instead of method 1's)

adi's code is imported read-only; the per-structure terms below re-run its own table / prior functions.
"""
import math
from adi import gate as G


def adi_terms(win, r, s):
    """Per adi structure (+ 'no gate' last): list of (log joint weight, P(win)) over flip times, exactly as
    adi.gate.win_probability combines them (same tables, structure prior, flip prior, cell prior)."""
    cfg = win._mcfg(s, r_type=r["machine_type"])
    taus, lprior = G._taus(r["cutoff"], cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(s, r["cutoff"], taus, win.ST, cfg["a"], cfg["b"], win._cp(r), cfg.get("half_life"))
    sp = win._sp(s); out = []
    for si, rows in enumerate(tb):
        if sp[si] <= 0:
            out.append([]); continue
        ls = math.log(sp[si])
        out.append([(ll + ls + lprior[ti], pr) for ti, (ll, pr) in enumerate(rows)])
    return out


def lag_prior_rows(win, sizes):
    """For each adi structure (+ 'no gate'): its size-lag prior pi_s(k)."""
    g = sizes.prior; c = sizes.couple_mix; idx = {tuple(st): i for i, st in enumerate(sizes.wST)}
    rows = []
    for form, lags in list(win.ST) + [("none", ())]:
        i = len(sizes.wST) if form == "none" else idx.get(tuple(lags)) if form in ("xor", "xs") else None
        cond = sizes.cond[i] if i is not None else g
        rows.append([(1 - c) * x + c * y for x, y in zip(cond, g)])
    return rows


def joint_parts(win, sizes, rows, r, s):
    """Everything the joint model needs for one query, computed once (then any ablation is cheap)."""
    terms = adi_terms(win, r, s); t = r["machine_type"]
    tabs = sizes._tables(s, sizes.mu[t], t); ll = [x[0] for x in tabs]; m = max(ll); L = [math.exp(x - m) for x in ll]
    logE = [math.log(max(1e-300, sum(p * l for p, l in zip(row, L)))) for row in rows]
    flat = [(j, pr, si) for si, tr in enumerate(terms) for j, pr in tr]
    z0 = G.logsumexp([j for j, _, _ in flat]); post_s = [0.0] * len(terms)
    for j, _, si in flat:
        post_s[si] += math.exp(j - z0)
    return dict(flat=flat, z0=z0, logE=logE, post_s=post_s, tabs=tabs, K=len(L))


def assemble(sizes, rows, r, s, P, feedback=True, size_from_adi=True, beta=1.0):
    flat, z0 = P["flat"], P["z0"]
    if feedback:
        z1 = G.logsumexp([j + beta * P["logE"][si] for j, _, si in flat])
        pw = sum(math.exp(j + beta * P["logE"][si] - z1) * pr for j, pr, si in flat)
        post = [0.0] * len(P["post_s"])
        for j, _, si in flat:
            post[si] += math.exp(j + beta * P["logE"][si] - z1)
    else:
        pw = sum(math.exp(j - z0) * pr for j, pr, _ in flat); post = P["post_s"]
    if size_from_adi:   # uses P(s | w) (no size feedback) -> P(k | w, y) exactly, no double counting of y
        prior = [sum(P["post_s"][si] * rows[si][k] for si in range(len(rows))) for k in range(P["K"])]
        sz = sizes.predict_with_prior(r, s, prior, P["tabs"])
    else:
        sz = sizes.predict(r, s)
    return [1 - pw] + [pw * x for x in sz], post


def joint_predict(win, sizes, rows, r, s, feedback=True, size_from_adi=True, beta=1.0):
    """Returns ([p0, p1, p2, p5, p20], structure posterior) under the joint model; flags = ablations."""
    return assemble(sizes, rows, r, s, joint_parts(win, sizes, rows, r, s), feedback, size_from_adi, beta)
