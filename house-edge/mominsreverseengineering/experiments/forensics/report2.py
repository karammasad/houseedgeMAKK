import pickle, zlib, math, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "experiments/oracle")
from common.data import public_dir, load_split
from adi import gate as G
R = pickle.load(open("experiments/forensics/rows.pkl", "rb")); P = [r for r in R if r["kind"] == "pseudo"]
pub = public_dir("../data/main"); q, seq = load_split(pub, "train"); mid2q = {r["machine_id"]: r for r in q}
def clus(rows, val):
    e = defaultdict(lambda: [0.0, 0])
    for r, v in zip(rows, val): e[r["ep"]][0] += v; e[r["ep"]][1] += 1
    N = sum(c for _, c in e.values()); m = sum(s for s, _ in e.values()) / N; return m, math.sqrt(sum((s - m * c) ** 2 for s, c in e.values())) / N, N
half = lambda r: zlib.crc32(r["ep"].encode()) % 2
# 1. PREFIX-ONLY gate strength (no information from the target or later ticks)
for r in P:
    s = seq[r["mid"]]; u = r["u"]
    if r["pair"] is None: r["pson"] = r["psoff"] = None; continue
    i, j = r["pair"]; n = np.zeros(2); k = np.zeros(2)
    for t in range(max(i, j), u):
        if s[t] < 0 or s[t - i] < 0 or s[t - j] < 0: continue
        g = int(s[t - i] > 0) ^ int(s[t - j] > 0)
        if r["tau"] is not None and t >= r["tau"]: g ^= 1
        n[g] += 1; k[g] += s[t] > 0
    rt = (k + .5) / (n + 1); r["pson"], r["psoff"] = float(rt.max()), float(rt.min())
print("1. residual (y_win - p) vs PREFIX-ONLY gate strength (history before the target only)")
res = lambda rows: np.array([float(r["y"] > 0) - r["p"] for r in rows])
for nm, key, edges in (("off-rate", "psoff", ((0, .03), (.03, .08), (.08, .2), (.2, 1.01))), ("on-rate", "pson", ((0, .5), (.5, .7), (.7, .85), (.85, 1.01)))):
    for lo, hi in edges:
        rows = [r for r in P if r[key] is not None and lo <= r[key] < hi]
        if len(rows) < 100: continue
        m, se, n = clus(rows, res(rows)); h = [clus([r for r in rows if half(r) == hh], res([r for r in rows if half(r) == hh]))[0] for hh in (0, 1)]
        print(f"   prefix {nm} [{lo},{min(hi,1)}) n={n:6d} resid {m:+.4f} (z {m/se:+.1f}) halves {h[0]:+.4f}/{h[1]:+.4f}")
# 2. flip timing vs condition / type / pair
M = list({r["mid"]: r for r in P}.values())
print("\n2. flip time (production posterior mean tau / L) by condition on long train machines")
for c in ("stable", "drift", "shared_change"):
    t = np.array([m["tau"] / m["L"] for m in M if m["cond"] == c and m["tau"] is not None])
    print(f"   {c:14s} flipped {len(t):3d}/{sum(1 for m in M if m['cond']==c)}  tau/L quantiles 0/10/50/90/100%: {np.round(np.quantile(t,[0,.1,.5,.9,1]),3)}")
print("   within-episode spread of tau (ticks): observed std vs permuted across episodes of the same condition")
rng = np.random.default_rng(0)
for c in ("drift", "shared_change"):
    mm = [m for m in M if m["cond"] == c and m["tau"] is not None]; eps = defaultdict(list)
    for m in mm: eps[m["ep"]].append(m["tau"])
    obs = np.mean([np.std(v) for v in eps.values() if len(v) >= 3]); allt = np.array([m["tau"] for m in mm]); sz = [len(v) for v in eps.values() if len(v) >= 3]
    nul = []
    for _ in range(500):
        p = rng.permutation(allt); i = 0; ss = []
        for z in sz: ss.append(np.std(p[i:i + z])); i += z
        nul.append(np.mean(ss))
    print(f"   {c:14s} {len(sz)} episodes: mean within-episode std {obs:.1f} vs null {np.mean(nul):.1f} (null 5-95%: {np.quantile(nul,.05):.1f}-{np.quantile(nul,.95):.1f})")
# 3. shared-tau Bayes factor per episode (marginal posterior vectors from the production gate)
from experiments_flip import posterior
import pickle as pk
gm = pk.load(open("experiments/gm_sb.pkl", "rb")); print("\n3. SHARED flip time within an episode? log Bayes factor of 'all 8 machines share one tau (or all none)' vs 'independent', per episode")
bf = defaultdict(list); epc = {}
for r in q: epc[r["episode_id"]] = None
cond = {x["query_id"]: x["condition"] for x in __import__("csv").DictReader(open("../data/main/train/conditions.csv"))}
by = defaultdict(list)
for r in q: by[r["episode_id"]].append(r)
import warnings; warnings.filterwarnings("ignore")
for ep, rs in sorted(by.items()):
    vecs = []
    for r in rs:
        s = seq[r["machine_id"]]; L = len(s); Pf, pf, pn, p, taus, mt = posterior(gm, s, L, r["machine_type"], r["signal"])
        cfg = gm._mcfg(s, r_type=r["machine_type"]); tt, lp = G._taus(L, cfg["p_change"], cfg["step"], cfg["tau_min"]); pr = np.exp(lp)
        post = np.concatenate([mt, [1 - Pf]]); vecs.append((post, pr))
    pr = vecs[0][1]; Lg = sum(np.log(np.maximum(v[0], 1e-300) / v[1]) for v in vecs)
    m = Lg.max(); bf[cond[rs[0]["query_id"]]].append(float(m + np.log((pr * np.exp(Lg - m)).sum())))
    if len({len(v[0]) for v in vecs}) > 1: bf[cond[rs[0]["query_id"]]][-1] = float("nan")
for c, v in bf.items(): v = np.array(v); print(f"   {c:14s} {len(v)} episodes: median log BF {np.nanmedian(v):+.1f}  share with log BF > 3: {np.nanmean(v>3):.2f}  (0 = no evidence for sharing)")
