"""adi's gate + GateSizes, but every learned prior is fit on the HISTORIES of train + validation + test.
Owner: Person D. Allowed by AGENTS.md ("fit may read pub/train fully, and the HISTORIES (not labels/targets) of other
splits"); no label, target or condition file is read. Same code path as adi.model.Model.fit; only the (queries, seq)
pool differs. `pool` lists the splits, `max_per_group` optionally caps machines per (split-agnostic) type to bound
runtime (deterministic: every k-th machine in sorted query order).
"""
import os
from adi import gate as G
from adi.model import Model as AdiModel, P_CHANGE_GRID, is_short
from common.data import load_split


def pooled_split(pub, pool, max_per_type=None):
    q, seq = [], {}
    for s in pool:
        if os.path.exists(os.path.join(pub, s, "history.csv")):
            a, b = load_split(pub, s); q += a; seq.update(b)
    if max_per_type:
        by = {}
        for r in sorted(q, key=lambda r: r["query_id"]):
            by.setdefault(r["machine_type"], []).append(r)
        q = [r for v in by.values() for r in v[:: max(1, -(-len(v) // max_per_type))]]
    return q, seq


class PooledAdi(AdiModel):
    def __init__(self, pool=("train", "validation", "test"), max_per_type=None, **kw):
        super().__init__(**kw); self.pool, self.max_per_type = tuple(pool), max_per_type

    def fit(self, pub):
        q, seq = pooled_split(pub, self.pool, self.max_per_type)
        p = self._raw_prior(q, seq)
        if self.prune > 0:
            keep = [i for i in range(len(self.ST)) if p[i] >= self.prune]
            self.ST = [self.ST[i] for i in keep]; p = [p[i] for i in keep] + [p[-1]]
        self.sprior = self._finish_prior(p)
        if self.cell_prior == "learn":
            self.cellpriors = G.fit_cell_priors(q, seq, self.ST, self.sprior, self.cfg, self.cell_group)
        if self.relearn and self.cellpriors is not None:
            p = G.learn_structure_prior(q, seq, self.ST, self.cfg["a"], self.cfg["b"], self.cfg["p_change"],
                                        tau_min=self.cfg["tau_min"], cellprior=self.cellpriors[()])
            self.sprior = self._finish_prior(p)
        if self.p_change_short == "learn":
            short = [r for r in q if is_short(seq[r["machine_id"]])]
            def ev(pc):
                cfg = dict(self.cfg, p_change=pc, step=max(4, self.cfg["step"]))
                return sum(G.win_probability(r, seq[r["machine_id"]], self.ST, self.sprior, evidence=True,
                                             cellprior=self._cp(r), **cfg)[1] for r in short)
            self.p_change_short = max(P_CHANGE_GRID, key=ev)
        self.sizes.fit(q, seq)
        return self
