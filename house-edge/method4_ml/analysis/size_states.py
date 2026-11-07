"""Are true size distributions discrete states? (analysis only)  python -m method4_ml.analysis.size_states"""
import csv
from collections import Counter, defaultdict
from common.data import load_split
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
q, seq = load_split(PUB, "validation")
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
d = defaultdict(Counter); dc = defaultdict(Counter)
for r in q:
    t = T[r["query_id"]]; w = 1 - t[0]; s = tuple(round(x / w, 3) for x in t[1:])
    d[r["machine_type"]][s] += 1; dc[(r["machine_type"], C[r["query_id"]])][s] += 1
for k, v in d.items():
    print(k, len(v), "distinct;", v.most_common(6))
for k in sorted(dc): print(k, len(dc[k]), dc[k].most_common(3))
# win prob values
pw = Counter(round(1 - T[r["query_id"]][0], 3) for r in q)
print("distinct P(win):", len(pw), pw.most_common(15))
