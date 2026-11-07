"""Do machines of one episode share the same win-gate structure? (train, long histories, MAP structure per machine)"""
import csv, math, itertools as it
from collections import Counter, defaultdict
from common.data import load_split
from method4_ml.model import Model
from adi import gate as G
PUB = "data/main"; gm = Model(blend=0).fit(PUB).gm
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
best = {}
for r in q:
    s = seq[r["machine_id"]]
    if sum(v < 0 for v in s) > len(s) / 2: continue
    tb = G.machine_tables(s, len(s), [], gm.ST, gm.cfg["a"], gm.cfg["b"], gm._cp(r))
    sc = [tb[i][0][0] + math.log(max(gm.sprior[i], 1e-12)) for i in range(len(gm.ST))]
    best[r["machine_id"]] = (gm.ST[max(range(len(sc)), key=sc.__getitem__)], r["episode_id"], C[r["query_id"]])
print("structures chosen:", Counter(v[0] for v in best.values()).most_common(6))
eps = defaultdict(list)
for m, (st, e, c) in best.items(): eps[e].append(st)
same = tot = 0
for v in eps.values():
    for a, b in it.combinations(v, 2): tot += 1; same += a == b
cnt = Counter(v[0] for v in best.values()); n = sum(cnt.values())
chance = sum((c / n) ** 2 for c in cnt.values())
print(f"same-structure rate within episode {same / tot:.3f} vs chance {chance:.3f}  ({tot} pairs)")
for cond in ("stable", "drift", "shared_change"):
    es = defaultdict(list)
    for m, (st, e, c) in best.items():
        if c == cond: es[e].append(st)
    s2 = t2 = 0
    for v in es.values():
        for a, b in it.combinations(v, 2): t2 += 1; s2 += a == b
    print(f"  {cond}: {s2 / t2:.3f} ({t2} pairs)")
