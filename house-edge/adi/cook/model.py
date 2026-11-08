"""adi "cook": adi.sizewin (round 7) + a two-level hierarchical size-split cell model on SHORT logs (adi/cook/hier.py),
with the sub-cell concentration kappa chosen per machine type by train marginal likelihood. Standard library only.

  python -m common.evaluate --method adi.cook --data data/main
  kwargs: kappa  "learn" (default) | number | {type: number or None}   (short logs)
          kappa_long  None (default, adi's independent sub-cells on long logs) | number | dict
          base   method2_sizes FINAL-style dict for the sizewin base (default: FINAL + short-only hotness prior)
          inner  "sizewin" (main model, default) | "r2" (adi.r2's widened round-two model, BASE_R2 + short hotness)
          size_menu  True (default) | False: snap size predictions onto a two-vector menu per type (adi/cook/menu.py,
                     port of method4_ml v8); menu_k winning history ticks per machine, menu_splits histories used

fit():
  1. if kappa == "learn": fit adi's win gate alone (hierarchy off), then for every train machine type sum the log
     marginal likelihood of short train logs (real ones + two 40-recent/24-older masked copies of each long log)
     under kappa in KAPPAS (None = off) and keep the argmax per type. Train only.
  2. install the per-type kappa and fit adi.sizewin on top (its logistic correction refits under the new gate).
  3. if size_menu: learn the two size vectors per type from HISTORY ticks (size model on seq[:t], payout at t as the
     label) and, in predict, snap each forecast's size part [p1, p2, p5, p20] / P(win) onto that segment.
"""
import random
from multiprocessing import get_context
import adi.gate as G
from adi.cook import hier, menu as MENU, couple as COUPLE
from adi.sizewin.model import Model as SizeWin
import method2_sizes.round2 as R
from common.data import load_split

KAPPAS = (None, 3, 10, 30, 100, 300, 1000)
PC_GRID = (.05, .1, .2, .35, .5, .65, .8, .9, .95)
_M = {}


def default_base(inner="sizewin"):
    if inner == "r2":
        from adi.r2.model import BASE_R2
        b = dict(BASE_R2); b["win_kwargs"] = dict(BASE_R2["win_kwargs"], machine_shift="short")
        return b
    b = dict(R.FINAL); b["win_kwargs"] = dict(R.ADI, machine_shift="short")
    return b


def _inner_cls(inner):
    if inner == "r2":
        from adi.r2.model import Model as R2
        return R2
    return SizeWin


def _mask(r, s, seed):
    rng = random.Random(f"cook{seed}|{r['machine_id']}"); t = len(s)
    keep = set(range(t - 40, t)) | set(rng.sample(range(t - 40), 24))
    return [x if i in keep else -1 for i, x in enumerate(s)]


def _pc_job(args):
    r, s = args; m = _M["adi"]; out = []
    for pc in PC_GRID:
        cfg = dict(m._mcfg(s, step=4, r_type=r["machine_type"]), p_change=pc)
        out.append(G.win_probability(r, s, m.ST, m._sp(s), cellprior=m._cp(r), evidence=True, **cfg)[1])
    return out


def _ev_job(args):
    r, s = args; m = _M["adi"]; out = []
    for k in KAPPAS:
        hier.install(k, None)
        out.append(G.win_probability(r, s, m.ST, m._sp(s), cellprior=m._cp(r), evidence=True,
                                     **m._mcfg(s, r_type=r["machine_type"]))[1])
    hier.install(None)
    return r["machine_type"], out


class Model:
    def __init__(self, kappa="learn", kappa_long=None, base=None, copies=2, workers=None, inner="sizewin",
                 size_menu=True, menu_k=32, menu_splits=("train", "validation", "test"), size_couple=False,
                 p_change_long=None, **kw):
        self.kappa, self.kappa_long, self.copies, self.workers = kappa, kappa_long, copies, workers
        self.inner_name = inner
        self.size_menu, self.menu_k, self.menu_splits = size_menu, menu_k, tuple(menu_splits)
        self.p_change_long = p_change_long   # "learn": long-log flip prior by train evidence (main-round default 0.5)
        self.size_couple = size_couple   # "adi": GateSizes' size-lag coupling reads adi's win posterior (couple.py)
        self.base_kw = base if base is not None else default_base(inner)
        self.kw = kw

    @property
    def base(self):
        """The fitted method2_sizes base model (same attribute as adi.sizewin, for adi.r2.run's describe())."""
        return self.inner.base

    @property
    def coef(self):
        return self.inner.coef

    def _learn_kappa(self, pub):
        import os
        from adi.model import Model as Adi
        hier.install(None)
        wk = {k: v for k, v in self.base_kw["win_kwargs"].items() if k != "machine_shift"}
        _M["adi"] = Adi(**wk).fit(pub)
        q, seq = load_split(pub, "train"); jobs = []
        for r in q:
            s = seq[r["machine_id"]]
            if min(s) < 0:
                jobs.append((r, s))
            else:
                jobs.extend((r, _mask(r, s, c)) for c in range(self.copies))
        n = self.workers or min(os.cpu_count() or 1, 8)
        with get_context("fork").Pool(n) as pool:
            res = pool.map(_ev_job, jobs, chunksize=8)
        tot = {}
        for t, v in res:
            tot[t] = [a + b for a, b in zip(tot.get(t, [0.0] * len(KAPPAS)), v)]
        self.kappa_evidence = {t: {str(k): round(x - v[0], 2) for k, x in zip(KAPPAS, v)} for t, v in tot.items()}
        if self.p_change_long == "learn":   # prior P(flip) for long logs, by summed train log-evidence (train only)
            longs = [(r, seq[r["machine_id"]]) for r in q if min(seq[r["machine_id"]]) >= 0]
            with get_context("fork").Pool(n) as pool:
                ev = pool.map(_pc_job, longs, chunksize=4)
            tot_pc = [sum(e[i] for e in ev) for i in range(len(PC_GRID))]
            self.p_change_fit = PC_GRID[max(range(len(PC_GRID)), key=tot_pc.__getitem__)]
            self.p_change_evidence = {str(pc): round(x - max(tot_pc), 2) for pc, x in zip(PC_GRID, tot_pc)}
            self.base_kw = dict(self.base_kw, win_kwargs=dict(self.base_kw["win_kwargs"], p_change=self.p_change_fit))
        return {t: KAPPAS[max(range(len(KAPPAS)), key=v.__getitem__)] for t, v in sorted(tot.items())}

    def _install(self):
        hier.tag_adi(); hier.install(self.kappa_fit, self.kappa_long)
        COUPLE.install(self.size_couple == "adi")

    def fit(self, pub):
        self.kappa_fit = self._learn_kappa(pub) if self.kappa == "learn" else self.kappa
        self._install()
        if self.workers is not None:
            self.kw.setdefault("workers", self.workers)
        self.inner = _inner_cls(self.inner_name)(base=self.base_kw, **self.kw).fit(pub)
        self.menu = None
        if self.size_menu:
            import os
            n = self.workers or min(os.cpu_count() or 1, 8)
            COUPLE.STATE["bypass"] = True   # menu positions only need approximate sizes: skip the slower coupled posterior
            try:
                self.menu = MENU.fit_menu(pub, self.menu_splits, self.inner.base.sizes, self.menu_k, n)
            finally:
                COUPLE.STATE["bypass"] = False
        return self

    def predict(self, pub, split):
        self._install()
        out = self.inner.predict(pub, split)
        if not self.menu:
            return out
        from common.data import load_split
        types = {r["query_id"]: r["machine_type"] for r in load_split(pub, split)[0]}
        for qid, p in out.items():
            pw = 1 - p[0]; V = self.menu.get(types[qid])
            if V is None or pw <= 1e-12:
                continue
            sz = MENU.snap([x / pw for x in p[1:]], V)
            out[qid] = [p[0]] + [pw * x for x in sz]
        return out
