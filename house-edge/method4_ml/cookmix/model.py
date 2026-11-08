"""method4_ml "cookmix" (Person D): linear pool of method4_ml v8 and adi.cook, then the v8 size menu on the pooled sizes.

  v8        method4_ml.model (adi gate + signal-shifted priors, GateSizes, tree stacker + offset-logistic, learned size menu)
  adi.cook  adi.sizewin + two-level hierarchical size-split cells on SHORT logs (kappa per type by train evidence)
P(win) and the size distribution are pooled with weight w (fixed at 0.5 in advance, as adi rounds 10-11: not tuned on
validation). The pooled size vector is snapped onto v8's learned 2-vector menu (types the self-check kept).
adi.cook patches adi.gate in-process; it is removed before every v8 call so v8's gate is exactly the unpatched one.

  python -m common.evaluate --method method4_ml.cookmix --data data/main
"""
from adi.cook import hier
from adi.cook.model import Model as Cook
from method4_ml.model import Model as V8
from method4_ml.size_menu import snap
from common.data import load_split


class Model:
    def __init__(self, w=0.5, v8=None, cook=None, menu=True, workers=None):
        a = dict(v8 or {}); c = dict(cook or {})
        if workers is not None:
            a.setdefault("workers", workers)
        self.w, self.menu = w, menu; self.v8 = V8(**a); self.cook = Cook(**c)

    def fit(self, pub):
        hier.install(None); self.v8.fit(pub)
        self.cook.fit(pub); hier.install(None)
        return self

    def predict(self, pub, split):
        hier.install(None); pa = self.v8.predict(pub, split)
        pb = self.cook.predict(pub, split); hier.install(None)
        q, _ = load_split(pub, split); ty = {r["query_id"]: r["machine_type"] for r in q}
        out = {}
        for k in pa:
            p = [self.w * x + (1 - self.w) * y for x, y in zip(pa[k], pb[k])]
            pw = 1 - p[0]
            if self.menu and self.v8.menu is not None and pw > 0:
                s = snap([x / pw for x in p[1:]], self.v8.menu.get(ty[k])); p = [p[0]] + [pw * x for x in s]
            out[k] = p
        return out
