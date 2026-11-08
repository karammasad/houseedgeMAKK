"""Probe the generator directly with validation TRUTH (diagnosis only): is the win-rate support discrete? does signal pin theta?
are sizes an exact menu? are parameters tied across a machine's cells / within types?"""
import csv, numpy as np
from collections import defaultdict, Counter
PUB = "data/main"
T = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/validation/conditions.csv"))}
Q = {r["query_id"]: r for r in csv.DictReader(open(f"{PUB}/validation/queries.csv"))}
pw = {k: 1 - v[0] for k, v in T.items()}
print("1. support of true P(win): distinct values (rounded 1e-6):", len({round(x, 6) for x in pw.values()}), "of", len(pw))
for ty in ("steady", "balanced", "jackpot"):
    v = np.array(sorted(pw[k] for k in pw if Q[k]["machine_type"] == ty)); lg = np.log(v / (1 - v))
    c = Counter(np.round(v, 3)); top = c.most_common(3)
    print(f"  {ty}: n {len(v)}  repeats@1e-3 {sum(n for _, n in c.items() if n > 1)}  top {top}  logit quantiles {np.round(np.quantile(lg, [.05, .25, .5, .75, .95]), 2)}")
print("\n2. signal vs true P(win), within type; split into on (P>.5) / off cells")
for ty in ("steady", "balanced", "jackpot"):
    for side, f in (("on", lambda x: x > .5), ("off", lambda x: x <= .5)):
        out = []
        for s in ("-1", "0", "1"):
            v = np.array([pw[k] for k in pw if Q[k]["machine_type"] == ty and Q[k]["signal"] == s and f(pw[k])])
            if len(v) > 3: out.append(f"sig{s:>2}: n{len(v):3d} mean {v.mean():.3f} sd {v.std():.3f}")
        print(f"  {ty:8s} {side:3s} | " + " | ".join(out))
print("\n3. size distribution given a win: distinct vectors per type")
for ty in ("steady", "balanced", "jackpot"):
    vs = Counter(tuple(np.round(np.array(T[k][1:]) / (1 - T[k][0]), 4)) for k in T if Q[k]["machine_type"] == ty and T[k][0] < .999)
    print(f"  {ty}: {len(vs)} distinct, top {vs.most_common(3)}")
print("\n4. is the P(win) value itself a ratio of simple numbers? check fraction fit with denominators <= 100")
fr = 0
for x in pw.values():
    if any(abs(x * d - round(x * d)) < 1e-6 for d in range(1, 101)): fr += 1
print("  exactly k/d (d<=100):", fr, "of", len(pw))
print("\n5. raw true values, a few per type x signal (to eyeball structure)")
for ty in ("steady", "jackpot"):
    print(" ", ty, [round(pw[k], 5) for k in list(pw) if Q[k]["machine_type"] == ty][:12])
