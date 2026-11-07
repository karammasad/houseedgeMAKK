"""At a change point, does the SIZE-STATE mapping (payout category at the size lag -> P(1x|win), and -> win rate) also reverse?
Train, long machines with a confident change (drift/shared) vs stable machines split at a fake midpoint (control).
Size lag = win pair's smaller lag (method 2 coupling finding). For each machine: P(1x|win) after a 'big' at the size lag minus
P(1x|win) after a '1x/loss' at the size lag, pre vs post. Same-sign = mapping kept, opposite = mapping reversed."""
import csv, math, statistics as st
from collections import defaultdict, Counter
from common.data import load_split, wins
from method4_ml.model import Model
from adi import gate as G
PUB = "data/main"
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
gm = Model(blend=0).fit(PUB).gm
pairs = [i for i, (f, s_) in enumerate(gm.ST) if f == "xor" and len(s_) == 2]
out = defaultdict(Counter); dif = defaultdict(list)
for r in q:
    s = seq[r["machine_id"]]
    if sum(x < 0 for x in s) > len(s) / 2: continue
    cut = len(s); cfg = gm._mcfg(s, r_type=r["machine_type"])
    taus, lprior = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(s, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r))
    _, si, ti = max(((tb[i][t][0] + math.log(max(gm.sprior[i], 1e-12)) + lprior[t], i, t) for i in pairs for t in range(len(taus) + 1)))
    c = C[r["query_id"]]; tau = cut // 2 if c == "stable" else (taus[ti] if ti < len(taus) else None)
    if tau is None: continue
    k = min(gm.ST[si][1])
    def eff(lo, hi):
        g = {0: [0, 0], 1: [0, 0]}   # 1 = previous payout at lag k was big (>=2)
        for t in range(max(lo, k), hi):
            if s[t] > 0 and s[t - k] >= 0:
                g[int(s[t - k] >= 2)][int(s[t] == 1)] += 1
        if min(sum(g[0]), sum(g[1])) < 5: return None
        return (g[1][1] + .5) / (sum(g[1]) + 1) - (g[0][1] + .5) / (sum(g[0]) + 1)
    a, b = eff(0, tau), eff(tau, cut)
    if a is None or b is None or abs(a) < .15: continue          # need a clear pre-change size gate
    out[c]["kept" if a * b > 0 else "reversed"] += 1; dif[c].append(b / a)
for c in ("stable", "drift", "shared_change"):
    print(f"{c:14s} size-gate direction after change: {dict(out[c])}   median post/pre effect ratio {st.median(dif[c]):+.2f}")
