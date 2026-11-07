"""Shape of true P(win) and links to size state / signal (analysis only)."""
import csv, math
from collections import Counter, defaultdict
from common.data import load_split
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
q, seq = load_split(PUB, "validation")
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
lg = Counter(); hi = {}
for r in q:
    t = T[r["query_id"]]; w = 1 - t[0]; lg[round(math.log(w / (1 - w)))] += 1
    # size state: 1 if more 2x than type mean "low" state
    hi[r["query_id"]] = t[2] / w > {"balanced": .45, "jackpot": .28, "steady": .3}[r["machine_type"]]
print("logit(P(win)) histogram:", sorted(lg.items()))
ct = Counter((r["signal"], hi[r["query_id"]]) for r in q); print("signal x size-state:", sorted(ct.items()))
ct = Counter((1 - T[r["query_id"]][0] > .5, hi[r["query_id"]]) for r in q); print("win-high x size-state:", sorted(ct.items()))
# episode: share of size state within episode
for c in ("stable", "shared_change", "drift"):
    fr = defaultdict(list)
    for r in q:
        if C[r["query_id"]] == c: fr[r["episode_id"]].append(hi[r["query_id"]])
    print(c, "per-episode # in high size state:", sorted(Counter(sum(v) for v in fr.values()).items()))
