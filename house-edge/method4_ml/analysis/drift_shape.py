"""Do drift episodes show intermediate (ramp) true win rates? (diagnosis: validation truth)"""
import csv
from collections import defaultdict
from common.data import load_split
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
T = {r["query_id"]: 1 - float(r["p_0"]) for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
q, _ = load_split(PUB, "validation")
for typ, lo, hi in (("steady", .35, .8), ("balanced", .3, .65)):
    print(typ, f"share of true P(win) in ({lo}, {hi}) by condition:")
    for c in ("stable", "short_history", "drift", "shared_change"):
        v = [T[r["query_id"]] for r in q if r["machine_type"] == typ and C[r["query_id"]] == c]
        mid = sum(lo < x < hi for x in v)
        print(f"   {c:14s} n={len(v):3d} mid-range {mid:2d} ({mid / len(v):.0%})   sorted: " + " ".join(f"{x:.2f}" for x in sorted(v)[len(v)//2 - 4: len(v)//2 + 8]))
