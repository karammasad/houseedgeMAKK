import pickle
import numpy as np
from collections import defaultdict
R = pickle.load(open("experiments/pooling/rows.pkl", "rb")); SZ = (1, 2, 5, 20)
def err(x, k):
    p = x[k]; oh = np.array([x["y"] == 0] + [x["y"] == c for c in SZ], float); return float(((np.concatenate([[1 - p], p * x["sz"]]) - oh) ** 2).sum()) * 1e4
for x in R: x["e_prod"], x["e_pool"], x["e_or"] = err(x, "p_prod"), err(x, "p_pool"), err(x, "p_or")
rng = np.random.default_rng(0)
def ci(rows, a, b):
    d = defaultdict(lambda: [0.0, 0])
    for x in rows: d[x["ep"]][0] += x[a] - x[b]; d[x["ep"]][1] += 1
    s = np.array([v[0] for v in d.values()]); c = np.array([v[1] for v in d.values()], float); pick = rng.integers(0, len(s), (3000, len(s)))
    m = np.sort(s[pick].sum(1) / c[pick].sum(1)); return s.sum() / c.sum(), m[75], m[2924], len(s)
bins = [("before flip (rel<0)", lambda x: x["rel"] < 0), ("0-10 after", lambda x: 0 <= x["rel"] < 10), ("10-20 after", lambda x: 10 <= x["rel"] < 20),
        ("20-40 after", lambda x: 20 <= x["rel"] < 40), ("QUERY-LIKE (>=38 after, as at the cutoff)", lambda x: x["rel"] >= 38), ("all rows", lambda x: True)]
print(f"{len(R)} rows, {len({x['ep'] for x in R})} shared_change episodes, {len({x['mid'] for x in R})} machines; five-category squared error x1e-4 vs realized payout")
print(f"{'rows':42s}{'n':>6s}{'prod':>9s}{'oracle':>9s}{'pooled':>9s}{'oracle-prod [95% CI, episodes]':>34s}{'pooled-prod [95% CI]':>28s}")
for nm, f in bins:
    rs = [x for x in R if f(x)]
    if not rs: continue
    o = ci(rs, "e_or", "e_prod"); p = ci(rs, "e_pool", "e_prod")
    print(f"{nm:42s}{len(rs):6d}{np.mean([x['e_prod'] for x in rs]):9.1f}{np.mean([x['e_or'] for x in rs]):9.1f}{np.mean([x['e_pool'] for x in rs]):9.1f}{o[0]:+10.2f} [{o[1]:+.2f},{o[2]:+.2f}]{p[0]:+14.2f} [{p[1]:+.2f},{p[2]:+.2f}]")
q = [x for x in R if x["rel"] >= 38]
print("\nquery-like rows: machine-level and episode-level view of the oracle gain")
dm = defaultdict(list); de = defaultdict(list)
for x in q: dm[x["mid"]].append(x["e_or"] - x["e_prod"]); de[x["ep"]].append(x["e_or"] - x["e_prod"])
m = np.array([np.mean(v) for v in dm.values()]); e = np.array([np.mean(v) for v in de.values()])
print(f"  machines n={len(m)}: mean change {m.mean():+.2f}; share of machines improved >1e-4: {np.mean(m<-1):.3f}, worsened >1e-4: {np.mean(m>1):.3f}; largest single-machine change {m.min():+.1f}")
print(f"  episodes n={len(e)}: mean change {e.mean():+.2f}, episode-mean range [{e.min():+.2f},{e.max():+.2f}]")
print(f"  mean |p_or - p_prod| {np.mean([abs(x['p_or']-x['p_prod']) for x in q]):.5f}, max {max(abs(x['p_or']-x['p_prod']) for x in q):.4f}")
print("\nby machine type (query-like): oracle - prod")
for t in ("steady", "balanced", "jackpot"):
    rs = [x for x in q if x["type"] == t]; o = ci(rs, "e_or", "e_prod"); print(f"  {t:9s} n={len(rs)} {o[0]:+.2f} [{o[1]:+.2f},{o[2]:+.2f}]")
