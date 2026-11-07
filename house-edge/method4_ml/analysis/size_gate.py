"""Does a win's size depend on past outcomes (lags / XOR pairs, own or neighbours)? (analysis on train)"""
import itertools as it, math
from collections import Counter, defaultdict
from common.data import load_split, wins
PUB = "data/main"
q, seq = load_split(PUB, "train")
def ll(groups):
    tot = 0
    for c in groups.values():
        n = sum(c)
        tot += sum(x * math.log((x + .5) / (n + 2)) for x in c if x)
    return tot
STRS = [(l,) for l in range(1, 13)] + list(it.combinations(range(1, 9), 2))
gain = Counter(); best = Counter(); nm = 0
for r in q:
    s = seq[r["machine_id"]]; w = wins(s)
    if sum(v > 0 for v in s) < 60: continue
    nm += 1
    T = [t for t in range(12, len(s)) if s[t] > 0]
    base = ll({0: [sum(s[t] == v for t in T) for v in (1, 2, 5, 20)]})
    top = (2.0, None)
    for st in STRS:
        g = defaultdict(lambda: [0, 0, 0, 0])
        for t in T:
            if any(w[t - l] < 0 for l in st): key = -1
            else:
                key = 0
                for l in st: key ^= w[t - l]
            g[key][(1, 2, 5, 20).index(s[t])] += 1
        d = ll(g) - base - 3 * (len(g) - 1) / 2 * math.log(max(len(T), 2))  # BIC
        gain[st] += d > 0
        if d > top[0]: top = (d, st)
    best[top[1]] += 1
print(nm, "long machines; # machines where structure beats no-structure by BIC:")
print(sorted(gain.items(), key=lambda x: -x[1])[:10])
print("best:", best.most_common(8))
# previous payout size dependence
g = defaultdict(lambda: [0, 0, 0, 0])
for r in q:
    s = seq[r["machine_id"]]; prev = None
    for t in range(len(s)):
        if s[t] > 0:
            if prev is not None: g[(r["machine_type"], prev)][(1, 2, 5, 20).index(s[t])] += 1
            prev = s[t]
for k in sorted(g): n = sum(g[k]); print(k, [round(x / n, 3) for x in g[k]], n)
