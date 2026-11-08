"""Signal-aware cell priors for adi's change-point gate. Owner: Person D.

Finding (analysis/onrate_links.py): within a machine type, `signal` correlates ~ +0.5 with the machine's gate "on"
rate (and on/off move together). adi's learned prior is per type only; its per type x signal variant re-fits the
strengths in every small group (noisy: it scored lower). Here strengths (k_on, k_off) are fit once per TYPE, as in
adi, and only the on/off MEANS are then fit per (type, signal) by marginal likelihood, strengths held fixed.
Learned on histories only (pool of splits); nothing hard-coded. Plugs into adi.model.Model via cellpriors keys
(type,) and (type, signal) with cell_group="type_signal".
"""
from adi import gate as G
from method4_ml.pooled import PooledAdi, pooled_split


def fit_means(queries, seq, ST, sprior, cfg, k_on, k_off, start, step=4, rounds=2):
    """Coordinate ascent over (on mean, off mean) with fixed strengths; objective = summed log marginal likelihood."""
    kw = dict(cfg); kw["step"] = max(step, cfg.get("step", 2))
    def ev(mo, mf):
        cp = (mo * k_on, (1 - mo) * k_on, mf * k_off, (1 - mf) * k_off)
        return sum(G.win_probability(r, seq[r["machine_id"]], ST, sprior, cellprior=cp, evidence=True, **kw)[1]
                   for r in queries)
    mo, mf = start; cur = ev(mo, mf)
    for _ in range(rounds):
        for v in G.MEANS_ON:
            if v != mo and (e := ev(v, mf)) > cur:
                mo, cur = v, e
        for v in G.MEANS_OFF:
            if v != mf and (e := ev(mo, v)) > cur:
                mf, cur = v, e
    return mo, mf


class SignalAdi(PooledAdi):
    """adi gate (+ GateSizes) with per type x signal cell-prior means. `pool` as in PooledAdi (default: train only)."""
    def __init__(self, pool=("train",), sig_min=20, **kw):
        super().__init__(pool=pool, **kw); self.sig_min = sig_min

    def fit(self, pub):
        super().fit(pub)                      # structure prior, per-type cell priors (strengths + means), sizes
        q, seq = pooled_split(pub, self.pool, self.max_per_type)
        pri = dict(self.cellpriors)
        for t in sorted({r["machine_type"] for r in q}):
            a1, b1, a0, b0 = pri.get((t,), pri[()]); k_on, k_off = a1 + b1, a0 + b0
            for s in (-1, 0, 1):
                grp = [r for r in q if r["machine_type"] == t and r["signal"] == s]
                if len(grp) < self.sig_min:
                    continue
                mo, mf = fit_means(grp, seq, self.ST, self.sprior, self.cfg, k_on, k_off, (a1 / k_on, a0 / k_off))
                pri[(t, s)] = (mo * k_on, (1 - mo) * k_on, mf * k_off, (1 - mf) * k_off)
        self.cellpriors, self.cell_group = pri, "type_signal"
        return self
