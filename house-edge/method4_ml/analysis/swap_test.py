"""Is a change a pure SWAP of the two cell rates (adi's assumption)? Train histories only (conditions used for slicing).
Per long machine: MAP plain XOR-pair structure and MAP flip time from the gate posterior. Orient cells so 'on' = higher
pre-change rate. Under a pure swap: rate(on cell, pre) == rate(cell that is 'on' after the swap, post), same for off.
z = (post - pre) / SE; var(z) ~ 1 if pure swap, > 1 if rates change at the flip. Control: stable machines at fake mid."""
import csv, math, statistics as st
from common.data import load_split, wins
from method4_ml.model import Model
from adi import gate as G
PUB = "data/main"
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
gm = Model(blend=0).fit(PUB).gm
pairs = [i for i, (f, st_) in enumerate(gm.ST) if f == "xor" and len(st_) == 2]
res = {}
for r in q:
    s = seq[r["machine_id"]]
    if sum(x < 0 for x in s) > len(s) / 2: continue
    cut = len(s); cfg = gm._mcfg(s, r_type=r["machine_type"])
    taus, lprior = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(s, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r))
    best = max(((tb[i][ti][0] + math.log(max(gm.sprior[i], 1e-12)) + lprior[ti], i, ti) for i in pairs for ti in range(len(taus) + 1)))
    _, si, ti = best; lags = gm.ST[si][1]; w = wins(s); cond = C[r["query_id"]]
    tau = taus[ti] if ti < len(taus) else None
    if cond == "stable": tau = cut // 2          # control: fake split, no real change
    elif tau is None: continue
    cnt = {(ph, x): [0, 0] for ph in (0, 1) for x in (0, 1)}
    for t in range(max(lags), cut):
        if w[t] < 0 or any(w[t - l] < 0 for l in lags): continue
        x = w[t - lags[0]] ^ w[t - lags[1]]; ph = int(t >= tau)
        if cond != "stable" and ph: x ^= 1                                   # undo the swap after the flip
        cnt[(ph, x)][0] += 1; cnt[(ph, x)][1] += w[t]
    for x in (0, 1):
        (n0, k0), (n1, k1) = cnt[(0, x)], cnt[(1, x)]
        if n0 < 8 or n1 < 8: continue
        p0, p1 = (k0 + .5) / (n0 + 1), (k1 + .5) / (n1 + 1); pp = (k0 + k1 + .5) / (n0 + n1 + 1)
        z = (p1 - p0) / math.sqrt(pp * (1 - pp) * (1 / n0 + 1 / n1) + 1e-9)
        role = "on" if (cnt[(0, x)][1] + .5) / (cnt[(0, x)][0] + 1) > (cnt[(0, 1 - x)][1] + .5) / (cnt[(0, 1 - x)][0] + 1) else "off"
        res.setdefault((cond, role), []).append((z, p1 - p0))
for k in sorted(res):
    zs = [a for a, _ in res[k]]; d = [b for _, b in res[k]]
    print(f"{k[0]:14s} {k[1]:3s} n={len(zs):3d}  var(z)={st.pvariance(zs):5.2f}  mean(post-pre)={st.mean(d):+.3f}  mean|post-pre|={st.mean(abs(x) for x in d):.3f}")
