"""Shared size-state shift for the size-split win cells (mominsreverseengineering; learns from histories only, stdlib only).

Hypothesis (FINDINGS Entry 4): one hidden binary size state moves the win logit by about +-gamma, with the same magnitude on the
on and off sides and a machine-specific direction, and adi's size-split cells treat the six sub-cell rates as exchangeable.
adi.gate._split_mixture now takes a 5th cell-prior element gamma (direction averaged 50/50 inside the marginal likelihood).
gamma is chosen PER MACHINE TYPE by the summed log marginal likelihood of the given histories (train, or any pool: no labels,
no validation targets), over a small grid that always contains 0. gamma = 0 for every type reproduces the current gate exactly,
so a mechanism that is absent in a new kit costs nothing: the data picks 0.
"""
from adi import gate as G

GAMMAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def attach_state_shift(gm, queries, seq, gammas=GAMMAS, step=4):
    """Append gamma to gm.cellpriors[(type, signal)]. Returns {type: (gamma, log-evidence gain vs gamma = 0)}."""
    chosen = {}; pri = dict(gm.cellpriors)
    for t in sorted({r["machine_type"] for r in queries}):
        ev = [0.0] * len(gammas)
        for r in queries:
            if r["machine_type"] != t:
                continue
            s = seq[r["machine_id"]]; cfg = gm._mcfg(s, step=step, r_type=t); base = tuple(gm._cp(r)[:4])
            for i, (_, z) in enumerate(G.win_probability_grid(r, s, gm.ST, gm._sp(s), [base + (g,) for g in gammas], **cfg)):
                ev[i] += z
        k = max(range(len(gammas)), key=ev.__getitem__)
        chosen[t] = (gammas[k], ev[k] - ev[0], [round(e - ev[0], 1) for e in ev])
        for sg in (-1, 0, 1):
            if (t, sg) in pri:
                pri[(t, sg)] = tuple(pri[(t, sg)][:4]) + (gammas[k],)
    gm.cellpriors = pri; gm.state_shift_fit = chosen
    return chosen
