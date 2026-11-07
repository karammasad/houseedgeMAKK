"""DIAGNOSTIC ONLY: HMM with the TRUE emissions (read from validation targets) - can history identify the state?"""
import csv, pickle
from collections import defaultdict
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.size_hmm import SizeHMM
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
vq, vs = load_split(PUB, "validation")
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
pred = pickle.load(open("method4_ml/analysis/m1_val.pkl", "rb"))
E = defaultdict(set)
for r in vq:
    t = T[r["query_id"]]; w = 1 - t[0]; E[r["machine_type"]].add(tuple(round(x / w, 4) for x in t[1:]))
E = {k: sorted(v, reverse=True) for k, v in E.items()}  # state 0 = more 1x
tune, _ = folds(PUB)
for h in (0, 0.001, 0.003, 0.01, 0.03):
    m = SizeHMM(h=h); m.E = {k: [list(a) for a in v] for k, v in E.items()}; m.pi0 = .45
    acc = defaultdict(list); out = {}
    for r in vq:
        p = m.state_prob(r, vs[r["machine_id"]]); t = T[r["query_id"]]; w = 1 - t[0]
        truth = int(round(t[1] / w, 4) != E[r["machine_type"]][0][0])
        acc[C[r["query_id"]]].append(p[truth])
        pw = 1 - pred[r["query_id"]][0]; out[r["query_id"]] = [1 - pw] + [pw * x for x in m.predict(r, vs[r["machine_id"]])]
    tp, rr = points(PUB, out, tune)
    print(f"h={h}: mean P(true state)", {c: round(sum(v) / len(v), 3) for c, v in acc.items()}, f"tuning {tp:.3f}")

# how often is the MAP history state (h=0) the true state, and how confident is it?
m = SizeHMM(h=0); m.E = {k: [list(a) for a in v] for k, v in E.items()}; m.pi0 = .45
agg = defaultdict(lambda: [0, 0, 0, 0])
for r in vq:
    p = m.state_prob(r, vs[r["machine_id"]]); t = T[r["query_id"]]; w = 1 - t[0]
    truth = int(round(t[1] / w, 4) != E[r["machine_type"]][0][0]); conf = max(p) > .95
    a = agg[(C[r["query_id"]], conf)]; a[0] += 1; a[1] += (p[1] > .5) == truth
for k in sorted(agg): print(k, "n", agg[k][0], "MAP correct", round(agg[k][1] / agg[k][0], 3))
