"""Analysis only (reads validation targets): error by condition x type, split into win and size parts."""
import csv, json, sys
from collections import defaultdict
from common.evaluate import folds, points, COLS
pub = "data/main"
P = json.load(open(sys.argv[1]))["validation"]
T = {r["query_id"]: r for r in csv.DictReader(open(f"{pub}/validation/targets.csv"))}
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{pub}/validation/conditions.csv"))}
Q = {r["query_id"]: r for r in csv.DictReader(open(f"{pub}/validation/queries.csv"))}
R = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(f"{pub}/validation/reference.csv"))}
tune, audit = folds(pub)
print("tuning %.3f audit %.3f full %.3f" % (points(pub, P, tune)[0], points(pub, P, audit)[0], points(pub, P)[0]))
agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for q, t in T.items():
    tr = [float(t[c]) for c in COLS]; p = P[q]; s = sum(p); p = [x / s for x in p]
    e = sum((a - b) ** 2 for a, b in zip(p, tr)); ref = sum((a - b) ** 2 for a, b in zip(R[q], tr))
    k = (C[q], Q[q]["machine_type"]); a = agg[k]
    a[0] += 1; a[1] += e; a[2] += ref; a[3] += (p[0] - tr[0]) ** 2 * 2
for k in sorted(agg):
    n, e, r, w = agg[k]; print("%-14s %-9s n=%3d mse %.5f ref %.5f ratio %.3f winpart %.2f" % (k + (n, e / n, r / n, e / r, w / e)))
