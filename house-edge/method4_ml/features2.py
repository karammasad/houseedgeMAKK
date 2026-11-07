"""Stacking features for Method 4 v2 (leakage-safe). Owner: Person D.

For a machine at tick t (prediction tick), every feature uses only seq_m[:t] of this machine and seq[:t] of the
other machines in the same episode. Training examples are history ticks t whose payout is observed (the label);
query examples use t = cutoff. All episode machines share ticks, so examples are built per (episode, t).

Feature groups
  m1_*   Method 1 (Bayesian change-point gate) posterior summaries: P(win), P(win | no change), P(change),
         P(no gate), top-structure mass, expected ticks since the change.
  ep_*   episode context from the OTHER machines at the same t: mean / max P(change), mean recency of change,
         mean m1 P(win) shift (full minus no-change).
  base   lag wins 1..8, recent win rates (10, 20, 40 observed ticks), ticks observed, missing share in last 40,
         machine type, signal.
"""
import math
from method1_gate.gate import machine_tables, logsumexp
from common.data import wins

TYPES = ("steady", "balanced", "jackpot")


def m1_summary(seq_trunc, ST, sprior, a=.5, b=.5, p_change=.5, step=2, tau_min=40):
    """Method 1 posterior summaries for predicting tick len(seq_trunc)."""
    cut = len(seq_trunc); taus = list(range(tau_min, cut - 2, step))
    lp = [math.log(p_change / len(taus))] * len(taus) + [math.log(1 - p_change)] if taus else [0.0]
    tb = machine_tables(seq_trunc, cut, taus, ST, a, b)
    J, PR, TI, SI = [], [], [], []
    for si, rows in enumerate(tb):
        ls = math.log(sprior[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lp[ti]); PR.append(pr); TI.append(ti); SI.append(si)
    z = logsumexp(J); W = [math.exp(j - z) for j in J]
    nt = len(taus)
    pw = sum(w * p for w, p in zip(W, PR))
    nochg = [(w, p) for w, p, ti in zip(W, PR, TI) if ti == nt]
    m = sum(w for w, _ in nochg); pw_nc = sum(w * p for w, p in nochg) / max(m, 1e-300)
    p_chg = 1 - m
    rec = sum(w * (cut - taus[ti]) for w, ti in zip(W, TI) if ti < nt) / max(p_chg, 1e-12) if p_chg > 1e-9 else cut
    smass = [0.0] * len(tb)
    for w, si in zip(W, SI):
        smass[si] += w
    return [pw, pw_nc, p_chg, smass[-1], max(smass[:-1]), rec / cut]


def base_features(seq_trunc):
    w = wins(seq_trunc); cut = len(w)
    f = [w[cut - l] if cut - l >= 0 else -1 for l in range(1, 9)]
    obs = [x for x in w if x >= 0]
    for k in (10, 20, 40):
        v = obs[-k:]; f.append((sum(v) + .5) / (len(v) + 1))
    f.append(len(obs)); f.append(sum(1 for x in w[-40:] if x < 0) / 40)
    return f


def episode_examples(machines, seq, ticks, ST, sprior, m1cfg):
    """machines: list of query dicts of one episode; ticks: prediction ticks to build.
    Returns {(machine_id, t): feature list}."""
    out = {}
    for t in ticks:
        M = {}
        for r in machines:
            M[r["machine_id"]] = m1_summary(seq[r["machine_id"]][:t], ST, sprior, **m1cfg)
        for r in machines:
            mid = r["machine_id"]; others = [M[o["machine_id"]] for o in machines if o["machine_id"] != mid]
            n = max(len(others), 1)
            ep = [sum(o[2] for o in others) / n, max((o[2] for o in others), default=0),
                  sum(o[5] for o in others) / n, sum(o[0] - o[1] for o in others) / n]
            m1 = M[mid]
            lg = math.log(max(m1[0], 1e-6) / max(1 - m1[0], 1e-6))
            out[(mid, t)] = m1 + [lg] + ep + base_features(seq[mid][:t]) + [TYPES.index(r["machine_type"]), r["signal"]]
    return out
