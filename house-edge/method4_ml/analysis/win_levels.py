"""What values does the true P(win) take, per type x signal x condition? (diagnosis, validation targets)"""
import csv
from collections import defaultdict, Counter
from common.data import load_split
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
T = {r["query_id"]: 1 - float(r["p_0"]) for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
q, seq = load_split(PUB, "validation")
d = defaultdict(list)
for r in q:
    if C[r["query_id"]] == "stable": d[(r["machine_type"], r["signal"])].append(T[r["query_id"]])
for k in sorted(d):
    v = sorted(d[k]); print(k, len(v), " ".join(f"{x:.2f}" for x in v))
