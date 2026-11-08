"""How much size error is left in WHICH of the 2 menu vectors applies (lambda)? Validation truth, diagnosis only."""
import csv, numpy as np
from collections import Counter
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"; tune, audit = folds(PUB)
T = {r["query_id"]: np.array([float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]) for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
Q = {r["query_id"]: r for r in csv.DictReader(open(f"{PUB}/validation/queries.csv"))}
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/validation/conditions.csv"))}
m = Model().fit(PUB); P = m.predict(PUB, "validation")
V = {}
for ty in ("steady", "balanced", "jackpot"):
    c = Counter(tuple(np.round(T[k][1:] / (1 - T[k][0]), 6)) for k in T if Q[k]["machine_type"] == ty); V[ty] = [np.array(v) for v, _ in c.most_common(2)]
rows = []
for k, p in P.items():
    p = np.array(p); s = p[1:] / (1 - p[0]); t = T[k][1:] / (1 - T[k][0]); V1, V2 = V[Q[k]["machine_type"]]; d = V1 - V2
    lt = float(np.clip((t - V2) @ d / (d @ d), 0, 1)); lp = float(np.clip((s - V2) @ d / (d @ d), 0, 1)); rows.append((k, lt, lp))
lt = np.array([r[1] for r in rows]); lp = np.array([r[2] for r in rows])
print("true lambda in {0,1}:", np.mean((lt < .01) | (lt > .99)))
print("pred lambda: share confident (<.1 or >.9) %.2f; accuracy of rounding %.3f; mean |err| %.3f" % (np.mean((lp < .1) | (lp > .9)), np.mean((lp > .5) == (lt > .5)), np.mean(np.abs(lp - lt))))
for lo, hi in ((0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)):
    sel = (lp >= lo) & (lp < hi); print(f"  pred lambda [{lo:.1f},{hi:.1f}) n={sel.sum():3d}  mean true {lt[sel].mean() if sel.any() else float('nan'):.3f}")
for cond in ("stable", "short_history", "drift", "shared_change"):
    sel = np.array([C[r[0]] == cond for r in rows]); print(f"  {cond:14s} mean |lambda err| {np.mean(np.abs(lp - lt)[sel]):.3f}")
def score_with(lam):
    out = {}
    for (k, _, _), l in zip(rows, lam):
        p = np.array(P[k]); V1, V2 = V[Q[k]["machine_type"]]; s = V2 + l * (V1 - V2); out[k] = [p[0]] + list((1 - p[0]) * s)
    return points(PUB, out, None)[0]
print("full: v8 %.3f | true V + pred lambda %.3f | true V + TRUE lambda (size oracle) %.3f" % (points(PUB, P, None)[0], score_with(lp), score_with(lt)))
