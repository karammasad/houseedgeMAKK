import pickle, zlib, math
from collections import defaultdict
import numpy as np
R = pickle.load(open("experiments/forensics/rows.pkl", "rb")); P = [r for r in R if r["kind"] == "pseudo"]; Lb = [r for r in R if r["kind"] == "label"]
def arr(rows, k): return np.array([r[k] for r in rows])
def clus(rows, val):
    """mean of val with episode-clustered SE"""
    e = defaultdict(lambda: [0.0, 0]); 
    for r, v in zip(rows, val): e[r["ep"]][0] += v; e[r["ep"]][1] += 1
    N = sum(c for _, c in e.values()); m = sum(s for s, _ in e.values()) / N
    se = math.sqrt(sum((s - m * c) ** 2 for s, c in e.values())) / N; return m, se, N
half = lambda r: zlib.crc32(r["ep"].encode()) % 2
res = lambda rows: arr(rows, "y") > 0
def resid_win(rows): return (arr(rows, "y") > 0).astype(float) - arr(rows, "p")
print(f"{len(P)} pseudo-query rows ({len({r['mid'] for r in P})} long machines, {len({r['ep'] for r in P})} episodes) + {len(Lb)} real train labels")
m, se, N = clus(P, resid_win(P)); print(f"overall mean win residual (y - p): {m:+.4f} +- {se:.4f}")
# 5-category reliability
print("\nA. five-category reliability, pseudo rows: mean predicted vs observed frequency")
p5 = np.array([r["p5"] for r in P]); y5 = np.array([[int(r["y"] == c) for c in (0, 1, 2, 5, 20)] for r in P]); print("  cat      pred    obs     diff")
for c, nm in enumerate(("0", "1", "2", "5", "20")): print(f"  {nm:5s} {p5[:,c].mean():8.4f}{y5[:,c].mean():8.4f}{y5[:,c].mean()-p5[:,c].mean():+9.4f}")
print("  win-prob reliability (bins of p):"); pw = arr(P, "p"); yw = (arr(P, "y") > 0).astype(float)
for lo, hi in ((0, .03), (.03, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .97), (.97, 1.01)):
    s = (pw >= lo) & (pw < hi)
    if s.sum(): mm, ss, n = clus([r for r, k in zip(P, s) if k], (yw - pw)[s]); print(f"   p in [{lo:.2f},{min(hi,1):.2f}) n={int(s.sum()):6d} pred {pw[s].mean():.3f} obs {yw[s].mean():.3f} resid {mm:+.4f} (z {mm/ss:+.1f})")
# slices
def slices():
    S = []
    for k in ("type", "cond", "sg", "form"): 
        for v in sorted({r[k] for r in P}, key=str): S.append((f"{k}={v}", lambda r, k=k, v=v: r[k] == v))
    for lo, hi in ((0, .4), (.4, .55), (.55, .7), (.7, .85), (.85, 1.01)): S.append((f"u/L [{lo},{hi})", lambda r, lo=lo, hi=hi: lo <= r["u"] / r["L"] < hi))
    for lo, hi in ((0, 20), (20, 60), (60, 120), (120, 999)): S.append((f"time since flip [{lo},{hi})", lambda r, lo=lo, hi=hi: r["tau"] is not None and lo <= r["u"] - r["tau"] < hi))
    S.append(("no flip (machine)", lambda r: r["tau"] is None))
    for lo, hi in ((0, .5), (.5, .7), (.7, .85), (.85, 1.01)): S.append((f"gate on-rate [{lo},{hi})", lambda r, lo=lo, hi=hi: r["son"] is not None and lo <= r["son"] < hi))
    for lo, hi in ((0, .03), (.03, .08), (.08, .2), (.2, 1)): S.append((f"gate off-rate [{lo},{hi})", lambda r, lo=lo, hi=hi: r["soff"] is not None and lo <= r["soff"] < hi))
    for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01)): S.append((f"top-structure mass [{lo},{hi})", lambda r, lo=lo, hi=hi: lo <= r["smass"] < hi))
    for v in (0, 1, 2, 5, 20): S.append((f"last payout={v}", lambda r, v=v: r["last"] == v)); S.append((f"payout at size lag={v}", lambda r, v=v: r["lagk"] == v))
    for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 999)): S.append((f"ticks since last >=5x [{lo},{hi})", lambda r, lo=lo, hi=hi: lo <= r["since_big"] < hi))
    for pr in sorted({r["pair"] for r in P if r["pair"]}): S.append((f"pair={pr}", lambda r, pr=pr: r["pair"] == pr))
    return S
print("\nB. residual (y_win - p) slices; cluster-robust z; replication = same sign in both episode halves with |z|>1.5 each. Rows with |z|>=3:")
out = []; S = slices()
for nm, f in S:
    rows = [r for r in P if f(r)]
    if len(rows) < 150: continue
    m, se, n = clus(rows, resid_win(rows)); h = [clus([r for r in rows if half(r) == hh], resid_win([r for r in rows if half(r) == hh])) for hh in (0, 1)]
    rep = all(abs(x[0] / x[1]) > 1.5 and np.sign(x[0]) == np.sign(m) for x in h)
    out.append((abs(m / se), nm, n, m, se, rep, h))
print(f"  {len(out)} slices tested; expected |z|>=3 by chance ~{len(out)*0.0027:.1f}")
for z, nm, n, m, se, rep, h in sorted(out, reverse=True)[:14]:
    print(f"  {nm:34s} n={n:6d} resid {m:+.4f} (z {m/se:+.1f}) halves {h[0][0]:+.4f}/{h[1][0]:+.4f} {'REPLICATES' if rep else ''}")
# size calibration given win
print("\nC. size distribution given a win: predicted vs observed, by last payout (payout size -> next sizes)")
W = [r for r in P if r["y"] > 0]
for v in (0, 1, 2, 5, 20):
    rows = [r for r in W if r["last"] == v]
    if len(rows) < 100: continue
    pr = np.array([r["p5"][1:] / r["p5"][1:].sum() for r in rows]).mean(0); ob = np.array([[int(r["y"] == c) for c in SZ] for r in rows for SZ in [(1, 2, 5, 20)]]).mean(0)
    print(f"  last={v:2d} n={len(rows):5d} pred {np.round(pr,3)} obs {np.round(ob,3)}")
# D: episode-shared residuals
print("\nD. episode-shared residual structure: variance of episode-mean residual at the same tick vs permutation null (machines shuffled across episodes, same tick)")
bytick = defaultdict(list)
for r in P: bytick[(r["u"], r["L"])].append(r)
rng = np.random.default_rng(0); obs_v = []; null_v = []
for key, rows in bytick.items():
    if len(rows) < 16: continue
    e = defaultdict(list)
    for r in rows: e[r["ep"]].append(float(r["y"] > 0) - r["p"])
    v = np.array([np.mean(x) for x in e.values() if len(x) >= 3])
    if len(v) < 4: continue
    obs_v.append(v.var()); allr = np.array([float(r["y"] > 0) - r["p"] for r in rows]); sizes = [len(x) for x in e.values() if len(x) >= 3]
    nv = []
    for _ in range(100): pm = rng.permutation(allr); i = 0; vv = []; 
    # simple null: split randomly into groups of the same sizes
    for _ in range(100):
        pm = rng.permutation(allr); i = 0; vv = []
        for sz in sizes: vv.append(pm[i:i + sz].mean()); i += sz
        nv.append(np.var(vv))
    null_v.append(np.mean(nv))
print(f"  {len(obs_v)} tick-groups: mean var of episode means observed {np.mean(obs_v):.5f} vs null {np.mean(null_v):.5f}  ratio {np.mean(obs_v)/np.mean(null_v):.3f} (1 = no shared structure)")
for cd in ("stable", "drift", "shared_change", "short_history"):
    pass
# E: flip structure
print("\nE. flip structure on long train machines (production posterior, flipped if P>0.5)")
M = {r["mid"]: r for r in P}.values(); M = list(M)
for k in ("type", "cond"):
    for v in sorted({m[k] for m in M}): 
        mm = [m for m in M if m[k] == v]; print(f"  {k}={v:14s} n={len(mm):4d} flipped {np.mean([m['tau'] is not None for m in mm]):.3f}  mean tau/L {np.mean([m['tau']/m['L'] for m in mm if m['tau']]):.3f}")
for pr in sorted({m["pair"] for m in M if m["pair"]}): 
    mm = [m for m in M if m["pair"] == pr]; print(f"  pair={pr} n={len(mm)} flipped {np.mean([m['tau'] is not None for m in mm]):.3f}")
print("  flipped share by condition x type:"); 
for c in ("stable", "drift", "shared_change"): print("   ", c, {t: round(np.mean([m['tau'] is not None for m in M if m['cond'] == c and m['type'] == t]), 2) for t in ("steady", "balanced", "jackpot")})
