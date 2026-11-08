"""Signal-dependent cell priors for adi's gate. Owner: Person D. Standard library only (uses adi.gate read-only).

Data finding (analysis/onrate_structure.py, validation truth): within (machine type, size state) the true on-cell win rate
shifts with the observed `signal` (jackpot +0.09 per signal step) and its spread inside a (type, state, signal) cell is only
~0.08-0.11 versus 0.164 for the type alone. adi's priors are per type and ignore signal, so each machine's rate is shrunk toward
the wrong centre. Here: keep adi's learned per-type prior and shift its on/off Beta means by beta_type * signal on the log-odds
scale (adi.gate.shift_cellprior). (mode, beta) is chosen per type by the marginal likelihood of the given histories (train, or any
pool of splits: histories only, no labels), over a small grid; beta = 0 is always a candidate, so no signal effect => no change.
"""
import math
from adi import gate as G

BETAS_ON = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8)
BETAS_OFF = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def shifted(cp, b_on, b_off, signal):
    """Shift the on-cell mean by b_on*signal and the off-cell mean by b_off*signal (log-odds), strengths kept."""
    return G.shift_cellprior(G.shift_cellprior(cp, b_on * signal, "on"), b_off * signal, "off")


def attach_signal_shift(gm, queries, seq, betas_on=BETAS_ON, betas_off=BETAS_OFF, step=4):
    """Fill gm.cellpriors[(type, signal)] and switch gm.cell_group to 'type_signal'.
    Returns {type: (beta_on, beta_off, log-evidence gain vs no shift)}."""
    combos = [(a, b) for a in betas_on for b in betas_off]; i0 = combos.index((0.0, 0.0))
    chosen = {}; pri = dict(gm.cellpriors)
    for t in sorted({r["machine_type"] for r in queries}):
        base = pri.get((t,), pri[()]); ev = [0.0] * len(combos)
        for r in queries:
            if r["machine_type"] != t:
                continue
            s = seq[r["machine_id"]]; cfg = gm._mcfg(s, step=step, r_type=t)
            cps = [shifted(base, a, b, r["signal"]) for a, b in combos]
            for i, (_, z) in enumerate(G.win_probability_grid(r, s, gm.ST, gm._sp(s), cps, **cfg)):
                ev[i] += z
        k = max(range(len(combos)), key=ev.__getitem__); a, b = combos[k]
        chosen[t] = (a, b, ev[k] - ev[i0])
        for sg in (-1, 0, 1):
            pri[(t, sg)] = shifted(base, a, b, sg)
    gm.cellpriors = pri; gm.cell_group = "type_signal"; gm.signal_shift = chosen
    return chosen
