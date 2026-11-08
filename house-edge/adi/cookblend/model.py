"""adi "cook" blend: 50/50 linear pool of adi.cook and adi.stack (standard library only). Owner: Adi.

  adi.cook   adi.sizewin + two-level hierarchical size-split cells on short logs, kappa per type by train evidence
  adi.stack  method4_ml v5's 770-tree stacker, exported and evaluated in pure Python (adi round 6)
P = w * cook + (1 - w) * stack over all five probabilities; w = 0.5 fixed in advance (as adi round 10).
Validation (main): tuning 58.294, audit 58.228, full 58.261.

method4_ml.model now imports numpy at module load (its v7 offset-logistic code). The v5 tree path used here never
calls numpy, so if numpy is not installed an empty stub module is registered before the import.

  python -m common.evaluate --method adi.cookblend --data data/main
"""
import sys, types
try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")
from adi.cook import hier
from adi.cook.model import Model as Cook
from adi.stack.model import Model as Stack


class Model:
    def __init__(self, w=0.5, cook=None, stack=None, workers=None):
        ck = dict(cook or {}); sk = dict(stack or {})
        ck.setdefault("size_menu", False); ck.setdefault("size_couple", False)  # the submitted main entry (CSV 8cc8b2a7) predates the size menu (round 12)
        if workers is not None:
            ck.setdefault("workers", workers); sk.setdefault("workers", workers)
        self.w = w; self.a = Cook(**ck); self.b = Stack(**sk)

    def fit(self, pub):
        self.a.fit(pub)
        hier.install(None)  # the stack's trees were trained on the unpatched gate: keep its features unchanged
        self.b.fit(pub)
        return self

    def predict(self, pub, split):
        pa = self.a.predict(pub, split)
        hier.install(None)
        pb = self.b.predict(pub, split)
        return {k: [self.w * x + (1 - self.w) * y for x, y in zip(pa[k], pb[k])] for k in pa}
