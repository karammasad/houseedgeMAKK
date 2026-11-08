"""Analysis only (reads validation targets): where is the remaining error? Oracle swaps of P(win) and sizes, by condition/type."""
import csv, json, sys
from collections import defaultdict
from common.evaluate import points, folds, COLS, CONDITIONS
pub = "data/main"; P = json.load(open(sys.argv[1]))["validation"]
T = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(f"{pub}/validation/targets.csv"))}
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{pub}/validation/conditions.csv"))}
Q = {r["query_id"]: r for r in csv.DictReader(open(f"{pub}/validation/queries.csv"))}
def mix(win_true, size_true, only=None):
    out = {}
    for q, p in P.items():
        t = T[q]; use = only is None or only(q)
        pw = 1 - (t[0] if (win_true and use) else p[0])
        s = [x / max(1 - t[0], 1e-12) for x in t[1:]] if (size_true and use) else [x / max(1 - p[0], 1e-12) for x in p[1:]]
        out[q] = [1 - pw] + [pw * x for x in s]
    return out
print("model           %.3f" % points(pub, P)[0])
print("true P(win)     %.3f" % points(pub, mix(1, 0))[0])
print("true sizes      %.3f" % points(pub, mix(0, 1))[0])
for c in CONDITIONS:
    print(f"true win+size only in {c:14s} %.3f" % points(pub, mix(1, 1, lambda q: C[q] == c))[0])
for ty in ("steady", "balanced", "jackpot"):
    print(f"true win only for {ty:9s} %.3f" % points(pub, mix(1, 0, lambda q: Q[q]["machine_type"] == ty))[0],
          f"| true sizes only for {ty:9s} %.3f" % points(pub, mix(0, 1, lambda q: Q[q]["machine_type"] == ty))[0])
for c in CONDITIONS:
    print(f"true win in {c:14s} %.3f" % points(pub, mix(1, 0, lambda q: C[q] == c))[0], f"| true sizes in {c:14s} %.3f" % points(pub, mix(0, 1, lambda q: C[q] == c))[0])
