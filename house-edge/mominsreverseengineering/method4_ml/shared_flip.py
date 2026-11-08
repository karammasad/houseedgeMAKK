"""Episode-shared change time. Owner: Person D. Standard library only (adi.gate read-only).

Finding (analysis/tau_sharing.py, train): in shared_change episodes ALL machines flip at the SAME tick (within-episode sd of the
posterior flip time 0.9 ticks vs 18.6 between episodes); in drift episodes they do not (12.7). The gate treats machines
independently, so a machine's flip time is inferred from its own ~10-25 informative post-flip ticks only.
Model per episode: H_sh = every machine flips at one common tau (uniform over the grid); H_ind = independent per-machine flips
(the existing gate). P(H_sh) = pi Z_sh / (pi Z_sh + (1-pi) Z_ind) from the episode's own histories; pi is learned on TRAIN
episodes by marginal likelihood (no condition labels). P(win) per machine = P(H_sh) * [sum_tau w_sh(tau) P_m(win|tau)] +
(1 - P(H_sh)) * P_ind. Machines of one episode share the cutoff (and the tau grid).
"""
import math
from adi import gate as G

PI_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def curves(gm, r, s):
    """Per tau index (flip at taus[i]; last index = no flip): log evidence summed over structures (with the structure prior) and
    the posterior-mean P(win | that tau). Also returns the machine's flip prior p_change and the tau grid."""
    cut = len(s); cfg = gm._mcfg(s, r_type=r["machine_type"])
    taus, _ = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"]); nt = len(taus)
    tb = G.machine_tables(s, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r), cfg.get("half_life")); sp = gm._sp(s)
    logE, Pw = [], []
    for ti in range(nt + 1):
        terms = [(rows[ti][0] + math.log(sp[si]), rows[ti][1]) for si, rows in enumerate(tb) if sp[si] > 0]
        z = G.logsumexp([a for a, _ in terms]); logE.append(z)
        Pw.append(sum(math.exp(a - z) * p for a, p in terms))
    return taus, logE, Pw, cfg["p_change"]


def episode_pw(gm, machines, seq, pi, cur=None):
    """{query_id: P(win)} for one episode's machines (queries at their cutoff; seq[m] = history before the cutoff).
    Returns also P(H_sh). `cur` can pass precomputed curves per machine."""
    cur = cur or {r["machine_id"]: curves(gm, r, seq[r["machine_id"]]) for r in machines}
    nt = len(next(iter(cur.values()))[0])
    # H_ind: product over machines of sum_tau prior_m(tau) E_m(tau); per-machine posterior-mean P(win) (existing gate)
    ind, logZi = {}, 0.0
    for r in machines:
        taus, logE, Pw, pc = cur[r["machine_id"]]
        lp = [math.log(pc / nt)] * nt + [math.log(1 - pc)]
        J = [a + b for a, b in zip(logE, lp)]; z = G.logsumexp(J); logZi += z
        ind[r["query_id"]] = sum(math.exp(j - z) * p for j, p in zip(J, Pw))
    # H_sh: one common tau (flip) for all machines
    S = [sum(cur[r["machine_id"]][1][ti] for r in machines) for ti in range(nt)]
    logZs = G.logsumexp(S) - math.log(nt); wz = G.logsumexp(S); w = [math.exp(x - wz) for x in S]
    sh = {r["query_id"]: sum(w[ti] * cur[r["machine_id"]][2][ti] for ti in range(nt)) for r in machines}
    if pi <= 0:
        return ind, 0.0, sh
    a = math.log(pi) + logZs; b = math.log(1 - pi) + logZi; m = max(a, b)
    psh = math.exp(a - m) / (math.exp(a - m) + math.exp(b - m))
    return {k: psh * sh[k] + (1 - psh) * ind[k] for k in ind}, psh, sh


def fit_pi(gm, queries, seq):
    """Mixture weight pi of the shared-flip hypothesis: argmax over PI_GRID of sum_episodes log(pi Z_sh + (1-pi) Z_ind) on the
    given (train) episodes, using histories only."""
    eps = {}
    for r in queries:
        eps.setdefault(r["episode_id"], []).append(r)
    zs, zi = [], []
    for ms in eps.values():
        cur = {r["machine_id"]: curves(gm, r, seq[r["machine_id"]]) for r in ms}
        nt = len(next(iter(cur.values()))[0]); li = 0.0
        for r in ms:
            taus, logE, Pw, pc = cur[r["machine_id"]]
            li += G.logsumexp([a + (math.log(pc / nt) if i < nt else math.log(1 - pc)) for i, a in enumerate(logE)])
        zi.append(li); zs.append(G.logsumexp([sum(cur[r["machine_id"]][1][ti] for r in ms) for ti in range(nt)]) - math.log(nt))
    best = None
    for pi in PI_GRID:
        if pi == 0:
            ll = sum(zi)
        else:
            ll = sum(G.logsumexp([math.log(pi) + a, math.log(1 - pi) + b]) for a, b in zip(zs, zi))
        if best is None or ll > best[0]:
            best = (ll, pi)
    gm.shared_pi = best[1]; return best[1]
