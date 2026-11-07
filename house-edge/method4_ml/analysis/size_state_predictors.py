"""DIAGNOSTIC: which observable features predict the TRUE size state at the cutoff? (validation targets)"""
import csv
from collections import defaultdict
from common.data import load_split, wins
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
vq, vs = load_split(PUB, "validation")
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
HI = {"steady": .6, "balanced": .36, "jackpot": .23}
st = {}
for r in vq:
    t = T[r["query_id"]]; w = 1 - t[0]; st[r["query_id"]] = int(t[1] / w < HI[r["machine_type"]])  # 1 = fewer 1x
def corr(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys)); sx = sum((a - mx) ** 2 for a in xs) ** .5; sy = sum((b - my) ** 2 for b in ys) ** .5
    return sxy / (sx * sy + 1e-12)
feats = defaultdict(list); ys = []
for r in vq:
    s = vs[r["machine_id"]]; cut = r["cutoff"]; w = wins(s); ys.append(st[r["query_id"]])
    ws = [v for v in s if v > 0]
    for k in (5, 10, 20, 40, 1000):
        last = ws[-k:]; feats[f"1x share last {k} wins"].append(sum(v == 1 for v in last) / max(len(last), 1))
    for l in range(1, 9): feats[f"win lag {l}"].append(max(w[cut - l], 0))
    feats["signal"].append(r["signal"]); feats["cutoff parity"].append(cut % 2)
    feats["win rate last 20"].append(sum(max(x, 0) for x in w[-20:]) / 20)
for k, v in feats.items(): print(f"{k:22s} corr with true size state {corr(v, ys):+.3f}")
