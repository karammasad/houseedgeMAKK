"""Do neighbours' past outcomes split a machine's win sizes? Compare to own-lag splits (train)."""
import math, random
from collections import defaultdict
from common.data import load_split, wins
q, seq = load_split("data/main", "train")
eps = defaultdict(list)
for r in q: eps[r["episode_id"]].append(r)
def gain(pairs):
    g = defaultdict(lambda: [0, 0]); n = [0, 0]
    for k, y in pairs:
        g[k][y] += 1
    tot = 0; all1 = sum(c[1] for c in g.values()); all0 = sum(c[0] for c in g.values()); N = all0 + all1
    if N == 0: return 0
    for c in g.values():
        m = sum(c); p = (c[1] + .5) / (m + 1); tot += c[1] * math.log(p) + c[0] * math.log(1 - p)
    p = (all1 + .5) / (N + 1); return tot - (all1 * math.log(p) + all0 * math.log(1 - p))
res = defaultdict(float); cnt = defaultdict(int)
for e, ms in eps.items():
    W = {r["machine_id"]: wins(seq[r["machine_id"]]) for r in ms}
    for r in ms:
        s = seq[r["machine_id"]]; T = [t for t in range(9, len(s)) if s[t] > 0]
        if len(T) < 40: continue
        for o in ms:
            wo = W[o["machine_id"]]; own = o is r
            for l in range(1, 9):
                pairs = [(wo[t - l], int(s[t] == 1)) for t in T if wo[t - l] >= 0]
                res[(own, l)] += gain(pairs); cnt[(own, l)] += 1
print("mean log-lik gain of 1x-vs-rest split per machine pair (null ~0.5):")
for own in (True, False):
    print(" own" if own else " neighbour", [round(res[(own, l)] / cnt[(own, l)], 2) for l in range(1, 9)])
# also by neighbour's payout being 1x vs other wins
