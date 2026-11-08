"""Feed adi's win-gate posterior into GateSizes' size-lag coupling (adi round 13). Standard library only.

GateSizes (method2_sizes) chooses which lag's payout category drives a machine's hidden size state, with a prior coupled to
the machine's win-gate structure (the size lag follows the win pair's first lag). It gets that win structure from its own
internal method-1 gate: flat Beta(.5, .5) cells and none of adi's learned type priors, signal slopes or short-log flip prior,
so on short logs its structure posterior is weak and the size lag stays uncertain (56-87% of short-log size error is
'state unsure', adi/FINDINGS round 13).

install() patches GateSizes in-process (fork-safe) so `_win_posterior` returns the posterior of the most recently fitted
adi gate model instead, mapped onto GateSizes' structure list (lag singles / pairs + 'none'): each adi structure contributes
its mass to the GateSizes structure with the same leading lags ("xs"/"xst" splits and AND pairs map to their XOR pair; lags
beyond GateSizes' range map to 'none'). The coupling table (P(size lag | win structure)) is still learned by GateSizes' own EM
on train, now under the sharper posterior. Train histories only; nothing about validation is read.
"""
import math
import adi.gate as G
import adi.model as AM
from method2_sizes.sizes import GateSizes

STATE = {"on": False, "adi": None, "cur": None, "reg": {}, "bypass": False}
_ORIG = {"wp": GateSizes._win_posterior, "fit": GateSizes.fit, "predict": GateSizes.predict, "adi_fit": AM.Model.fit}


def _adi_fit(self, pub):
    out = _ORIG["adi_fit"](self, pub)
    STATE["adi"] = self
    return out


def _fit(self, queries, seq):
    # machine lookup by list identity is only valid while `seq` is alive: clear it when fit returns, so a later list that
    # reuses a freed id can never pick up another machine's type/signal (that made runs non-deterministic)
    STATE["reg"] = {id(seq[r["machine_id"]]): (seq[r["machine_id"]], r) for r in queries}
    try:
        return _ORIG["fit"](self, queries, seq)
    finally:
        STATE["reg"] = {}


def _predict(self, q, seq_m):
    prev = STATE["cur"]; STATE["cur"] = q
    try:
        return _ORIG["predict"](self, q, seq_m)
    finally:
        STATE["cur"] = prev


def adi_structure_posterior(m, r, seq_m):
    """Posterior over m.ST + ['none'] for this history (flip times marginalized), using adi's learned priors."""
    cfg = m._mcfg(seq_m, r_type=r["machine_type"])
    taus, lp = G._taus(len(seq_m), cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(seq_m, len(seq_m), taus, m.ST, cfg["a"], cfg["b"], m._cp(r), cfg["half_life"])
    sp = m._sp(seq_m); ls = []
    for i, rows in enumerate(tb):
        ls.append(-1e300 if sp[i] <= 0 else math.log(sp[i]) + G.logsumexp([ll + x for (ll, _), x in zip(rows, lp)]))
    z = G.logsumexp(ls)
    return [math.exp(x - z) for x in ls]


def _target(self, form, lags):
    if form in ("xs", "xst") or form.startswith("and"):
        lags = lags[:2]
    key = tuple(lags[:2]) if len(lags) >= 2 else tuple(lags)
    idx = getattr(self, "_cook_widx", None)
    if idx is None:
        idx = self._cook_widx = {s: i for i, s in enumerate(self.wST)}
    return idx.get(key, len(self.wST))


def _win_posterior(self, seq_m):
    m = STATE["adi"]; hit = STATE["reg"].get(id(seq_m))
    r = hit[1] if hit is not None and hit[0] is seq_m else STATE["cur"]
    if STATE["bypass"] or m is None or r is None or len(seq_m) < 2:
        return _ORIG["wp"](self, seq_m)
    post = adi_structure_posterior(m, r, seq_m)
    out = [0.0] * (len(self.wST) + 1)
    for (form, lags), p in zip(m.ST, post[:-1]):
        out[_target(self, form, lags)] += p
    out[-1] += post[-1]
    z = sum(out)
    return [x / z for x in out]


def install(on=True):
    STATE["on"] = on
    GateSizes._win_posterior = _win_posterior if on else _ORIG["wp"]
    GateSizes.fit = _fit if on else _ORIG["fit"]
    GateSizes.predict = _predict if on else _ORIG["predict"]
    AM.Model.fit = _adi_fit if on else _ORIG["adi_fit"]
