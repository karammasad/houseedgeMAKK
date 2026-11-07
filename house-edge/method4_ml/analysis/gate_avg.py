"""Does averaging P(win) of gate variants beat adi alone? (tuning half; sizes fixed to the adi/GateSizes model)"""
import pickle
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"
m = Model(blend=0).fit(PUB); base = m.predict(PUB, "validation"); tune, _ = folds(PUB)
m1 = pickle.load(open("method4_ml/analysis/m1_val.pkl", "rb"))
for w in (0, .1, .2, .3, .5):
    out = {}
    for k, v in base.items():
        pw = (1 - w) * (1 - v[0]) + w * (1 - m1[k][0]); sz = [x / (1 - v[0]) for x in v[1:]]
        out[k] = [1 - pw] + [pw * x for x in sz]
    print(f"adi/M1 win-prob weight on M1 = {w}: tuning {points(PUB, out, tune)[0]:.3f}")
