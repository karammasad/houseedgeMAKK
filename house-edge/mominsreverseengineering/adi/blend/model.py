"""adi round 10: linear pool of two standard-library models (Owner: Adi).

For a squared-error (Brier-type) score the matching way to combine probability forecasts is the linear pool
(Neyman & Roughgarden 2021; Ranjan & Gneiting 2010). The two members are built very differently:
  adi.sizewin  team FINAL + 6-number logistic size-state correction (round 7, full 58.214)
  adi.stack    method4_ml v5's 770-tree stacker, exported and evaluated in pure Python (round 6, full 58.192)
P = w * sizewin + (1 - w) * stack, all five probabilities. Both members are standard library only.

  python -m common.evaluate --method adi.blend --data data/main --kwargs '{"w": 0.5}'
"""
from adi.sizewin.model import Model as SizeWin
from adi.stack.model import Model as Stack


class Model:
    def __init__(self, w=0.5, sizewin=None, stack=None):
        self.w = w; self.a = SizeWin(**(sizewin or {})); self.b = Stack(**(stack or {}))

    def fit(self, pub):
        self.a.fit(pub); self.b.fit(pub)
        return self

    def predict(self, pub, split):
        pa = self.a.predict(pub, split); pb = self.b.predict(pub, split)
        return {k: [self.w * x + (1 - self.w) * y for x, y in zip(pa[k], pb[k])] for k in pa}
