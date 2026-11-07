"""How do payout sizes move over time? (analysis only)  python -m method4_ml.analysis.sizes_time"""
import csv, statistics as st
from collections import defaultdict
from common.data import load_split

PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
q, seq = load_split(PUB, "validation")
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
SZ = (1, 2, 5, 20)
def emp(s, lo, hi, prior, k=4):
    c = [sum(1 for x in s[max(lo,0):hi] if x == v) for v in SZ]; n = sum(c)
    return [(a + k * p) / (n + k) for a, p in zip(c, prior)]
tm = defaultdict(lambda: [0]*4)
for r in q:
    t = T[r["query_id"]]; w = 1 - t[0]
    for i in range(4): tm[r["machine_type"]][i] += t[i+1] / w
cnt = defaultdict(int)
for r in q: cnt[r["machine_type"]] += 1
tm = {k: [x / cnt[k] for x in v] for k, v in tm.items()}
res = defaultdict(lambda: defaultdict(list))
for r in q:
    t = T[r["query_id"]]; w = 1 - t[0]; ts = [x / w for x in t[1:]]; s = seq[r["machine_id"]]; cut = r["cutoff"]; pr = tm[r["machine_type"]]
    for name, lo in [("all", 0), ("last80", cut-80), ("last40", cut-40), ("last20", cut-20)]:
        e = emp(s, lo, cut, pr); res[C[r["query_id"]]][name].append(sum((a-b)**2 for a, b in zip(e, ts)))
    res[C[r["query_id"]]]["typemean"].append(sum((a-b)**2 for a, b in zip(pr, ts)))
for c, d in res.items():
    print(c, {k: round(st.mean(v), 4) for k, v in d.items()})
# shared deviation: within shared_change episodes, correlation of (true size - typemean) across machines
print("\nPer-episode true size deviations from type mean (shared_change, first 4 episodes):")
eps = defaultdict(list)
for r in q:
    if C[r["query_id"]] == "shared_change":
        t = T[r["query_id"]]; w = 1 - t[0]
        eps[r["episode_id"]].append((r["machine_type"], [round(t[i+1]/w - tm[r["machine_type"]][i], 2) for i in range(4)], round(w, 2)))
for e in list(eps)[:4]:
    print(e); [print("   ", x) for x in eps[e]]
