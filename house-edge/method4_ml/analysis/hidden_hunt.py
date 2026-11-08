"""Hunt for hidden simulator structure in HISTORIES only (no targets). Each test prints z / p; look for |z| well above ~4."""
import numpy as np, math
from collections import defaultdict
from common.data import load_split
PUB = "data/main"
M = []   # (split, episode, type, signal, seq array)
for sp in ("train", "validation", "test"):
    q, seq = load_split(PUB, sp)
    for r in q: M.append((sp, r["episode_id"], r["machine_type"], r["signal"], np.array(seq[r["machine_id"]])))
print("machines", len(M), "ticks", sum(len(m[4]) for m in M))
W = lambda s: np.where(s < 0, -1, (s > 0).astype(int))
# A. win rate by tick mod k, per type (chi-square z)
print("\nA. tick mod k (max z over k=2..48 and residues, by type)")
for ty in ("steady", "balanced", "jackpot"):
    best = []
    for k in range(2, 49):
        c = np.zeros((k, 2))
        for m in M:
            if m[2] != ty: continue
            w = W(m[4]); t = np.arange(len(w)); ok = w >= 0
            for res in range(k):
                sel = ok & (t % k == res); c[res] += (w[sel].sum(), sel.sum())
        p = c[:, 0].sum() / c[:, 1].sum(); z = (c[:, 0] / c[:, 1] - p) / np.sqrt(p * (1 - p) / c[:, 1])
        best.append((np.abs(z).max(), k, int(np.abs(z).argmax())))
    print(f"  {ty}: top3 (|z|, k, residue)", sorted(best, reverse=True)[:3], " (expect ~3.5 by chance over ~1100 cells)")
# B. long-lag autocorrelation of win bits within machine (lags 13..80), centered per machine
print("\nB. win-bit autocorr lags 13..80 (pooled z), by type")
for ty in ("steady", "balanced", "jackpot"):
    zs = []
    for L in range(13, 81):
        num = den = 0.0; n = 0
        for m in M:
            if m[2] != ty: continue
            w = W(m[4]).astype(float); ok = w >= 0
            mu = w[ok].mean() if ok.sum() > 5 else 0
            a, b = w[L:], w[:-L]; v = (a >= 0) & (b >= 0)
            num += ((a[v] - mu) * (b[v] - mu)).sum(); den += v.sum() * 1.0; n += v.sum()
        vr = np.mean([(W(m[4])[W(m[4]) >= 0].var()) for m in M if m[2] == ty]) ** 1
        zs.append((num / max(den, 1) / vr * math.sqrt(n), L))
    print(f"  {ty}: top3 |z| (z, lag)", sorted(zs, key=lambda x: -abs(x[0]))[:3])
# C. same-tick and lagged cross-machine win correlation within episode (machine-mean centered)
print("\nC. cross-machine corr, episode-mates, lag L (m at t vs m' at t-L), pooled z")
eps = defaultdict(list)
for m in M: eps[(m[0], m[1])].append(m)
for L in (0, 1, 2, 3, 5, 8, 12):
    num = 0.0; n = 0; sx = 0.0
    for e, ms in eps.items():
        A = np.array([W(x[4]).astype(float) for x in ms if len(x[4]) == len(ms[0][4])])
        if len(A) < 2: continue
        mu = np.array([a[a >= 0].mean() if (a >= 0).sum() > 5 else 0 for a in A])
        for i in range(len(A)):
            for j in range(len(A)):
                if i == j or (L == 0 and i > j): continue
                a, b = A[i][L:], A[j][:len(A[j]) - L]; v = (a >= 0) & (b >= 0)
                da, db = a[v] - mu[i], b[v] - mu[j]; num += (da * db).sum(); sx += (da ** 2).sum() * 0 + math.sqrt((da ** 2).sum() * (db ** 2).sum()); n += v.sum()
    print(f"  L={L}: corr {num / sx:+.4f}  z~{num / sx * math.sqrt(n):+.1f}")
# F. pity timer: P(big | ticks since last big) and P(win | ticks since last 5x/20x), by type
print("\nF. hazard of 5x/20x vs ticks since last 5x/20x (observed ticks only), by type")
for ty in ("steady", "balanced", "jackpot"):
    cnt = defaultdict(lambda: [0, 0]); wcnt = defaultdict(lambda: [0, 0])
    for m in M:
        if m[2] != ty: continue
        last = None
        for t, x in enumerate(m[4]):
            if x < 0: continue
            if last is not None:
                g = min(t - last, 60) // 5; cnt[g][0] += x >= 5; cnt[g][1] += 1; wcnt[g][0] += x > 0; wcnt[g][1] += 1
            if x >= 5: last = t
    print(f"  {ty}: P(big) by gap/5:", " ".join(f"{cnt[g][0] / max(cnt[g][1], 1):.3f}" for g in range(13)), "| P(win):", " ".join(f"{wcnt[g][0] / max(wcnt[g][1], 1):.2f}" for g in range(13)))
# G. position effects: win rate vs distance to end of history, and vs absolute tick (by type)
print("\nG. win rate by absolute tick bucket (50 ticks) and by last-20 vs rest, by type")
for ty in ("steady", "balanced", "jackpot"):
    rate = defaultdict(lambda: [0, 0])
    for m in M:
        if m[2] != ty: continue
        w = W(m[4])
        for t in range(len(w)):
            if w[t] >= 0: rate[t // 50][0] += w[t]; rate[t // 50][1] += 1
    print(f"  {ty}:", " ".join(f"{rate[b][0] / rate[b][1]:.3f}" for b in sorted(rate) if rate[b][1] > 500))
# H. machine index within episode vs type / signal (assignment structure)
print("\nH. machine index vs type (counts) from query ids")
q, _ = load_split(PUB, "train"); c = defaultdict(int)
for r in q: c[(int(r["machine_id"].split("-m")[-1]), r["machine_type"])] += 1
for i in range(8): print("  m%d" % i, {t: c[(i, t)] for t in ("steady", "balanced", "jackpot")})
