"""What predicts a machine's on-rate (best XOR pair, stable, long train histories)? per machine type."""
import csv, itertools as it, math, statistics as st
from collections import defaultdict
from common.data import load_split, wins
PUB = "data/main"
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
def fitrates(w):
    best = None
    for L in it.combinations(range(1, 9), 2):
        c = [[0, 0], [0, 0]]
        for t in range(8, len(w)): c[w[t - L[0]] ^ w[t - L[1]]][w[t]] += 1
        ll = sum(k * math.log((k + .5) / (n0 + k + 1)) + n0 * math.log((n0 + .5) / (n0 + k + 1)) for n0, k in c)
        if best is None or ll > best[0]: best = (ll, c)
    c = best[1]; r = [(k + .5) / (n0 + k + 1) for n0, k in c]; ns = [n0 + k for n0, k in c]; hi = max((0, 1), key=lambda i: r[i])
    return r[hi], r[1 - hi], ns[hi], ns[1 - hi]
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b); sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb + 1e-12)
by = defaultdict(list)
for r in q:
    if C[r["query_id"]] != "stable": continue
    s = seq[r["machine_id"]]; w = wins(s); on, off, nh, nl = fitrates(w)
    wn = [v for v in s if v > 0]
    by[r["machine_type"]].append(dict(on=on, off=off, nh=nh, signal=r["signal"], share1=sum(v == 1 for v in wn) / max(len(wn), 1),
        share20=sum(v >= 5 for v in wn) / max(len(wn), 1), winrate=sum(v > 0 for v in s) / len(s), n_on=nh))
for t, v in by.items():
    print(t, len(v), "machines; on-rate sd %.3f" % st.pstdev(x["on"] for x in v))
    for k in ("signal", "off", "share1", "share20", "winrate", "n_on"):
        print(f"   corr(on-rate, {k:8s}) = {corr([x['on'] for x in v], [x[k] for x in v]):+.2f}")
