"""Ensemble of independently built models (selection on TUNING, audit shown as a check):
v7 (mine) + adi.sizewin (stdlib, adi gate + size-state logistic) + method2 FINAL (adi gate + joint-EM sizes).
Average the full 5-class probability vectors (convex combination: always a valid distribution)."""
import json, pickle, os, itertools, numpy as np
from common.evaluate import points, folds
from method4_ml.model import Model
import importlib
PUB = "data/main"; tune, audit = folds(PUB)
cache = "method4_ml/cache/ensemble_preds.pkl"
if os.path.exists(cache):
    preds = pickle.load(open(cache, "rb"))
else:
    import method2_sizes.round2 as R
    preds = {"v7": Model().fit(PUB).predict(PUB, "validation")}
    m2 = importlib.import_module("method2_sizes.model").Model(**R.FINAL).fit(PUB); preds["final"] = m2.predict(PUB, "validation")
    preds["sizewin"] = importlib.import_module("adi.sizewin.model").Model().fit(PUB).predict(PUB, "validation")
    pickle.dump(preds, open(cache, "wb"))
def avg(w):
    ks = preds["v7"].keys(); out = {}
    for k in ks:
        v = sum(w[n] * np.array(preds[n][k]) for n in w) / sum(w.values()); out[k] = list(v / v.sum())
    return out
def sc(p): return points(PUB, p, tune)[0], points(PUB, p, audit)[0], points(PUB, p, None)[0]
for n in preds: a = sc(preds[n]); print(f"{n:22s} {a[0]:.3f} {a[1]:.3f} {a[2]:.3f}")
print("\nweights (v7, sizewin, final)          tuning audit  full")
for w in [(1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (1, 0, 1), (3, 1, 1), (1, 2, 0)]:
    a = sc(avg(dict(v7=w[0], sizewin=w[1], final=w[2]))); print(f"{str(w):38s} {a[0]:.3f} {a[1]:.3f} {a[2]:.3f}")
