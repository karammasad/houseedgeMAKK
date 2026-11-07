"""Where is the v3 base model's error? Swap in true win prob / true sizes (analysis only, validation tuning half)."""
import csv, pickle, os
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
m = Model(blend=0).fit(PUB); pred = m.predict(PUB, "validation"); tune, _ = folds(PUB)
def mix(pw_src, sz_src):
    out = {}
    for k in pred:
        pw = 1 - pw_src[k][0]; sz = [x / max(1 - sz_src[k][0], 1e-12) for x in sz_src[k][1:]]
        out[k] = [1 - pw] + [pw * x for x in sz]
    return out
for name, p in [("model", pred), ("true P(win) + model sizes", mix(T, pred)), ("model P(win) + true sizes", mix(pred, T))]:
    tp, r = points(PUB, p, tune); print(f"{name:28s} tuning {tp:.3f}", {c: round(v, 4) for c, v in r.items()})
