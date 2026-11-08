"""Exact size menu (2 vectors per type, identical across machines): how far are v7's size predictions from the segment V1-V2,
and what would snapping them onto it be worth? Bound uses V's from validation truth; the 'learned' row uses V's estimated
from TRAIN data only (EM on train winning payouts as a 2-component mixture with per-machine weight)."""
import csv, numpy as np
from collections import Counter
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"; tune, audit = folds(PUB)
T = {r["query_id"]: np.array([float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]) for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
Q = {r["query_id"]: r for r in csv.DictReader(open(f"{PUB}/validation/queries.csv"))}
m = Model().fit(PUB); pred = m.predict(PUB, "validation")
V = {}
for ty in ("steady", "balanced", "jackpot"):
    c = Counter(tuple(np.round(T[k][1:] / (1 - T[k][0]), 6)) for k in T if Q[k]["machine_type"] == ty)
    V[ty] = [np.array(v) for v, _ in c.most_common(2)]
def project(s, V1, V2):
    d = V1 - V2; lam = np.clip((s - V2) @ d / (d @ d), 0, 1); return V2 + lam * d, lam
e_pred = e_proj = 0; off = []
snap = {}
for k, p in pred.items():
    p = np.array(p); s = p[1:] / (1 - p[0]); t = T[k][1:] / (1 - T[k][0]); V1, V2 = V[Q[k]["machine_type"]]
    sp, lam = project(s, V1, V2); e_pred += ((s - t) ** 2).sum(); e_proj += ((sp - t) ** 2).sum(); off.append(((s - sp) ** 2).sum())
    snap[k] = [p[0]] + list((1 - p[0]) * sp)
print(f"size-given-win SSE: v7 {e_pred:.4f}  snapped-to-true-segment {e_proj:.4f}  (off-segment part {np.sum(off):.4f})")
for nm, P in (("v7", pred), ("v7 sizes snapped (true V, bound)", snap)):
    print(f"{nm:34s} tuning {points(PUB, P, tune)[0]:.3f}  audit {points(PUB, P, audit)[0]:.3f}  full {points(PUB, P, None)[0]:.3f}")
# what V's does the fitted size model itself carry? compare v7 predictions' extreme points per type to the true V
for ty in ("steady", "balanced", "jackpot"):
    S = np.array([np.array(pred[k][1:]) / (1 - pred[k][0]) for k in pred if Q[k]["machine_type"] == ty])
    i, j = S[:, 0].argmax(), S[:, 0].argmin()
    print(f"  {ty}: true V1 {np.round(V[ty][0], 3)} V2 {np.round(V[ty][1], 3)} | pred extremes {np.round(S[i], 3)} {np.round(S[j], 3)}")
