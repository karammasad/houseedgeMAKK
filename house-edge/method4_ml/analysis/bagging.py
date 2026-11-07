"""Variance from the random choice of training ticks, and bagging over tick samples (tuning + audit + full)."""
import numpy as np
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"; tune, audit = folds(PUB); B = 5
preds = []
for b in range(B):
    m = Model(tick_seed=b).fit(PUB); p = m.predict(PUB, "validation"); preds.append(p)
    print(f"tick sample {b}: tuning {points(PUB, p, tune)[0]:.3f} audit {points(PUB, p, audit)[0]:.3f} full {points(PUB, p, None)[0]:.3f}", flush=True)
for nb in (3, 5):
    avg = {k: list(np.mean([p[k] for p in preds[:nb]], axis=0)) for k in preds[0]}
    print(f"bag of {nb}: tuning {points(PUB, avg, tune)[0]:.3f} audit {points(PUB, avg, audit)[0]:.3f} full {points(PUB, avg, None)[0]:.3f}")
