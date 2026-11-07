"""Where is Method 1's error? Swap in true win prob / true sizes (analysis only).
Run from house-edge/: python -m method4_ml.analysis.decompose"""
import csv, json, os, pickle, time
from common.data import load_split
from common.evaluate import points, folds
from method1_gate.model import Model as M1

PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
cache = "method4_ml/analysis/m1_val.pkl"
if os.path.exists(cache):
    pred = pickle.load(open(cache, "rb"))
else:
    t0 = time.time(); m = M1().fit(PUB); pred = m.predict(PUB, "validation"); print("m1 secs", time.time() - t0)
    pickle.dump(pred, open(cache, "wb"))
tune, audit = folds(PUB)
def mix(pw_src, sz_src):
    out = {}
    for k in pred:
        pw = 1 - pw_src[k][0]; sz = [x / max(1 - sz_src[k][0], 1e-12) for x in sz_src[k][1:]]
        out[k] = [1 - pw] + [pw * x for x in sz]
    return out
for name, p in [("method1", pred), ("true P(win) + m1 sizes", mix(T, pred)), ("m1 P(win) + true sizes", mix(pred, T))]:
    tp, r = points(PUB, p, tune); print(f"{name:28s} tuning {tp:.3f}", {c: round(v, 4) for c, v in r.items()})
