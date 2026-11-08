"""Stacking features for Method 4 v3 (leakage-safe). Owner: Person D. Standard library + adi/gate (read-only import).

For a machine at prediction tick t every feature uses only seq_m[:t] of this machine and seq[:t] of the other
machines of the same episode. Training examples are history ticks t whose payout is observed (the label);
query examples use t = cutoff.

Groups
  gate_*  summaries of a fitted change-point gate model `gm` (adi.model.Model: learned structure prior, learned
          per-type cell priors, short-log flip prior): P(win), P(win | no flip), P(flip), P(no gate), top-2
          structure masses, expected flip recency (works for plain, AND-family and size-split structures).
  ep_*    the OTHER machines' gate signals at the same t (mean / max P(flip), mean recency, mean win-shift).
  size_*  (optional) the size model's P(1x | win) for this machine at t and its deviation from the type mean: the
          hidden size state also moves the win rate (adi round 4), so the GBM can learn that shift.
  base    lag wins 1..8, recent win rates (10/20/40), observed count, missing share of last 40, type, signal.
  pol     (v9, optional: gm.polarity) polarity bank (polarity.py): strongest neighbour-lag, own big-payout-lag and
          own-XOR-neighbour evidence on this machine's own history, and the shift it predicts now. Appended LAST so the
          logit_stack column layout of the first 29 columns is unchanged.
Data-derived settings live on gm (set by model.fit, learned from the round's data): gm.type_order (machine types; the main
round's three keep their main indices, new types are appended), gm.short_obs (observed-tick count of a short log).
"""
import math
import numpy as np
from adi import gate as G
from common.data import wins

TYPES = ("steady", "balanced", "jackpot")     # default order only; the model sets gm.type_order from the data
SHORT_OBS = 64                                 # default only; the model sets gm.short_obs from the train short logs


def type_order(names):
    """Main-round types keep their main indices (results on main stay identical); any other type is appended in sorted order."""
    return list(TYPES) + sorted(set(names) - set(TYPES))


def type_index(gm, name):
    order = getattr(gm, "type_order", None) or list(TYPES)
    return order.index(name) if name in order else len(order)


def is_short(seq_trunc, short_obs=SHORT_OBS):
    return sum(1 for x in seq_trunc if x >= 0) <= short_obs


def gate_summary(seq_trunc, r, gm):
    """Gate posterior summaries for predicting tick len(seq_trunc)."""
    cut = len(seq_trunc); cfg = dict(gm.cfg)
    if gm.p_change_short is not None and is_short(seq_trunc, getattr(gm, "short_obs", SHORT_OBS)):
        cfg["p_change"] = gm.p_change_short
    taus = list(range(cfg["tau_min"], cut - 2, cfg["step"])); nt = len(taus)
    lp = [math.log(cfg["p_change"] / nt)] * nt + [math.log(1 - cfg["p_change"])] if nt else [0.0]
    tb = G.machine_tables(seq_trunc, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r))
    J, PR, TI, SI = [], [], [], []
    for si, rows in enumerate(tb):
        if gm.sprior[si] <= 0:
            continue
        ls = math.log(gm.sprior[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lp[ti]); PR.append(pr); TI.append(ti); SI.append(si)
    z = G.logsumexp(J); W = [math.exp(j - z) for j in J]
    pw = sum(w * p for w, p in zip(W, PR))
    nc = [(w, p) for w, p, ti in zip(W, PR, TI) if ti == nt]
    m = sum(w for w, _ in nc); pw_nc = sum(w * p for w, p in nc) / max(m, 1e-300); p_chg = 1 - m
    rec = (sum(w * (cut - taus[ti]) for w, ti in zip(W, TI) if ti < nt) / p_chg) if (p_chg > 1e-9 and nt) else cut
    mass = [0.0] * len(tb)
    for w, si in zip(W, SI):
        mass[si] += w
    order = sorted(range(len(tb) - 1), key=lambda i: -mass[i]); top = [mass[i] for i in order] + [0.0, 0.0]
    return [pw, pw_nc, p_chg, mass[-1], top[0], top[1], rec / cut]


def base_features(seq_trunc):
    w = wins(seq_trunc); cut = len(w)
    f = [w[cut - l] if cut - l >= 0 else -1 for l in range(1, 9)]
    obs = [x for x in w if x >= 0]
    for k in (10, 20, 40):
        v = obs[-k:]; f.append((sum(v) + .5) / (len(v) + 1))
    f.append(len(obs)); f.append(sum(1 for x in w[-40:] if x < 0) / 40)
    return f


def episode_examples(machines, seq, ticks, gm, sz=None):
    """machines: query dicts of one episode; ticks: prediction ticks. Returns {(machine_id, t): features}."""
    out = {}; pol = getattr(gm, "polarity", False)
    for t in ticks:
        PF = None
        if pol:
            from method4_ml.polarity import episode_polarity
            P = np.array([seq[r["machine_id"]][:t] for r in machines], dtype=np.int16)
            PF = episode_polarity(np.where(P < 0, -1, (P > 0)).astype(np.int8), P, t)
        M = {r["machine_id"]: gate_summary(seq[r["machine_id"]][:t], r, gm) for r in machines}
        for k, r in enumerate(machines):
            mid = r["machine_id"]; others = [M[o["machine_id"]] for o in machines if o["machine_id"] != mid]
            n = max(len(others), 1)
            ep = [sum(o[2] for o in others) / n, max((o[2] for o in others), default=0),
                  sum(o[6] for o in others) / n, sum(o[0] - o[1] for o in others) / n]
            g = M[mid]; lg = math.log(max(g[0], 1e-6) / max(1 - g[0], 1e-6))
            szf = []
            if sz is not None:   # size model's P(1x | win) at this tick (hidden size state) and its type-centred version
                a = sz.predict(r, seq[mid][:t])[0]; szf = [a, a - sz.mu.get(r["machine_type"], a)]
            out[(mid, t)] = g + [lg] + ep + base_features(seq[mid][:t]) + [type_index(gm, r["machine_type"]), r["signal"]] + szf + ([] if PF is None else PF[k].tolist())
    return out
