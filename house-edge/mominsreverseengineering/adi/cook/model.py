"""adi "cook": adi.sizewin (round 7) + a two-level hierarchical size-split cell model on SHORT logs (adi/cook/hier.py),
with the sub-cell concentration kappa chosen per machine type by train marginal likelihood. Standard library only.

  python -m common.evaluate --method adi.cook --data data/main
  kwargs: kappa  "learn" (default) | number | {type: number or None}   (short logs)
          kappa_long  None (default, adi's independent sub-cells on long logs) | number | dict
          base   method2_sizes FINAL-style dict for the sizewin base (default: FINAL + short-only hotness prior)
          inner  "sizewin" (main model, default) | "r2" (adi.r2's widened round-two model, BASE_R2 + short hotness)

fit():
  1. if kappa == "learn": fit adi's win gate alone (hierarchy off), then for every train machine type sum the log
     marginal likelihood of short train logs (real ones + two 40-recent/24-older masked copies of each long log)
     under kappa in KAPPAS (None = off) and keep the argmax per type. Train only.
  2. install the per-type kappa and fit adi.sizewin on top (its logistic correction refits under the new gate).
"""
import random
from multiprocessing import get_context
import adi.gate as G
from adi.cook import hier
from adi.sizewin.model import Model as SizeWin
import method2_sizes.round2 as R
from common.data import load_split

KAPPAS = (None, 3, 10, 30, 100, 300, 1000)
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


def _ev_job(args):
    r, s = args; m = _M["adi"]; out = []
    for k in KAPPAS:
        hier.install(k, None)
        out.append(G.win_probability(r, s, m.ST, m._sp(s), cellprior=m._cp(r), evidence=True,
                                     **m._mcfg(s, r_type=r["machine_type"]))[1])
    hier.install(None)
    return r["machine_type"], out


class Model:
    def __init__(self, kappa="learn", kappa_long=None, base=None, copies=2, workers=None, inner="sizewin", **kw):
        self.kappa, self.kappa_long, self.copies, self.workers = kappa, kappa_long, copies, workers
        self.inner_name = inner
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
        return {t: KAPPAS[max(range(len(KAPPAS)), key=v.__getitem__)] for t, v in sorted(tot.items())}

    def _install(self):
        hier.tag_adi(); hier.install(self.kappa_fit, self.kappa_long)

    def fit(self, pub):
        self.kappa_fit = self._learn_kappa(pub) if self.kappa == "learn" else self.kappa
        self._install()
        if self.workers is not None:
            self.kw.setdefault("workers", self.workers)
        self.inner = _inner_cls(self.inner_name)(base=self.base_kw, **self.kw).fit(pub)
        return self

    def predict(self, pub, split):
        self._install()
        return self.inner.predict(pub, split)
