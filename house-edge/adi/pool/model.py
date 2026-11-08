"""adi "pool": linear pool of method4_ml (Kareem's v8/v9 stacker, scikit-learn) and adi.cook (standard library). Owner: Adi.

For a squared-error score the matching combination of probability forecasts is the linear pool (Neyman & Roughgarden
2021). The two members are built differently (gradient-boosted stacker on gate features vs a fully Bayesian gate with a
hierarchical short-log prior, exact size menu and adi-coupled size state), so their errors are partly independent.
P = w * method4 + (1 - w) * cook over all five probabilities; w = 0.5 fixed in advance (no weight tuning).

Isolation: adi.cook patches adi.gate / GateSizes in-process. method4 is always fitted and predicted with those patches
OFF (fit order: method4 first; before every method4 call the patches are uninstalled), so its features are exactly
those of a standalone method4 run.

Meant to ship as a saved checkpoint (adi/ckpt_entry.py): fitted once, pickled, and predict.py only loads + predicts.

  python -m common.evaluate --method adi.pool --data data/main          (needs the pinned scikit-learn stack)
"""
from adi.cook import hier, couple
from adi.cook.model import Model as Cook


def _patches_off():
    hier.install(None); couple.install(False)


class Model:
    def __init__(self, w=0.5, m4=None, cook=None, workers=None):
        self.w = w; self.workers = workers
        self.m4_kw = dict(m4 or {}); self.m4_kw["cache"] = False      # /submission is read-only on the runner
        self.cook_kw = dict(cook if cook is not None else {"inner": "r2", "size_couple": "adi"})
        if workers is not None:
            self.m4_kw.setdefault("workers", workers); self.cook_kw.setdefault("workers", workers)

    @classmethod
    def from_members(cls, m4, cook, w=0.5):
        """Pool two ALREADY-FITTED members (the morning runbook fits each candidate once and reuses it)."""
        self = cls.__new__(cls); self.w = w; self.workers = None; self.m4_kw = {}; self.cook_kw = {}
        self.m4 = m4; self.cook = cook
        return self

    def fit(self, pub):
        from method4_ml.model import Model as M4
        _patches_off()
        self.m4 = M4(**self.m4_kw).fit(pub)
        self.cook = Cook(**self.cook_kw).fit(pub)
        _patches_off()
        return self

    def set_workers(self, n):
        """Used by the checkpoint predict.py: cap parallelism (results do not depend on it)."""
        self.m4.workers = n
        self.cook.workers = n
        for o in (self.cook, getattr(self.cook, "inner", None)):
            if o is not None and hasattr(o, "workers"):
                o.workers = n

    def predict(self, pub, split):
        _patches_off()
        a = self.m4.predict(pub, split)
        b = self.cook.predict(pub, split)      # installs its own patches
        _patches_off()
        return {k: [self.w * x + (1 - self.w) * y for x, y in zip(a[k], b[k])] for k in a}
