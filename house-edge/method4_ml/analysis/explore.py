"""Data exploration for Method 4 (analysis only; reads validation targets, never used by the model).
Run from house-edge/: python -m method4_ml.analysis.explore"""
import csv, os, statistics as st
from collections import defaultdict, Counter
from common.data import load_split, wins, PAYOUTS

PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

q, seq = load_split(PUB, "validation")
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
print("conditions:", Counter(C.values()))
print("types x signal:", Counter((r["machine_type"], r["signal"]) for r in q))
pw = [1 - T[r["query_id"]][0] for r in q]
hist = Counter(round(x, 1) for x in pw); print("true P(win) histogram (0.1 bins):", sorted(hist.items()))
# size dist given win by type
by = defaultdict(list)
for r in q:
    t = T[r["query_id"]]; s = 1 - t[0]
    by[r["machine_type"]].append([x / s for x in t[1:]])
for k, v in by.items():
    print(k, "mean size dist|win:", [round(st.mean(x[i] for x in v), 3) for i in range(4)],
          "sd:", [round(st.pstdev(x[i] for x in v), 3) for i in range(4)])
# history coverage
for c in ("stable", "short_history", "drift", "shared_change"):
    obs = [sum(v >= 0 for v in seq[r["machine_id"]]) for r in q if C[r["query_id"]] == c]
    print(c, "observed ticks: min", min(obs), "median", st.median(obs), "max", max(obs))
# by signal: P(win)
for s in (-1, 0, 1):
    v = [1 - T[r["query_id"]][0] for r in q if r["signal"] == s]
    print("signal", s, "mean true P(win)", round(st.mean(v), 3))
