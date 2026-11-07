"""Do machines in an episode share win-rate levels beyond type x signal? (train histories, stable episodes)
For each long stable machine: best XOR-pair structure by likelihood; on-rate (higher cell) and off-rate (lower cell).
Residual after removing the type x signal mean, then ICC across machines in the same episode (same-type pairs and all pairs)."""
import csv, itertools as it, math, statistics as st
from collections import defaultdict
from common.data import load_split, wins
PUB = "data/main"
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
def rates(w):
    best = None
    for L in it.combinations(range(1, 9), 2):
        c = [[0, 0], [0, 0]]
        for t in range(8, len(w)): c[w[t - L[0]] ^ w[t - L[1]]][w[t]] += 1
        ll = sum(k * math.log((k + .5) / (n0 + k + 1)) + n0 * math.log((n0 + .5) / (n0 + k + 1)) for n0, k in c)
        if best is None or ll > best[0]: best = (ll, c)
    c = best[1]; r = [(k + .5) / (n0 + k + 1) for n0, k in c]; ns = [n0 + k for n0, k in c]
    hi = max((0, 1), key=lambda i: r[i]); lo = 1 - hi
    return r[hi], r[lo], ns[hi], ns[lo]
rows = []
for r in q:
    if C[r["query_id"]] != "stable": continue
    on, off, nh, nl = rates(wins(seq[r["machine_id"]]))
    rows.append((r["episode_id"], r["machine_type"], r["signal"], on, off, nh, nl))
grp = defaultdict(list)
for e, t, s, on, off, nh, nl in rows: grp[(t, s)].append((on, off))
mean = {k: (st.mean(x[0] for x in v), st.mean(x[1] for x in v)) for k, v in grp.items()}
for idx, name in ((3, "on-rate"), (4, "off-rate")):
    res = defaultdict(list)
    for row in rows: res[row[0]].append(row[idx] - mean[(row[1], row[2])][idx - 3])
    allv = [x for v in res.values() for x in v]; tot = st.pvariance(allv)
    # pairwise within-episode covariance of residuals
    cov = [a * b for v in res.values() for a, b in it.combinations(v, 2)]
    print(f"{name}: residual sd {tot ** .5:.3f}; mean within-episode pair product {st.mean(cov):.5f}; implied ICC {st.mean(cov) / tot:+.3f} (0 = no sharing)")
