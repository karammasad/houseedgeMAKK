"""Does the WIN gate depend on payout category (loss/1x/big) and not just win/loss? Held-out test on STABLE train
machines: structure + cell rates chosen on first 70% of ticks, scored on last 30% (per-spin log-lik).
Compared per machine: best XOR pair of win bits (2 cells) / best 2-lag category pair (9 cells) /
best XOR pair + secondary single-lag category split inside each XOR cell."""
import csv, itertools as it, math
from common.data import load_split
PUB = "data/main"
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
cat = lambda v: 0 if v == 0 else (1 if v == 1 else 2)
def fitll(cells_tr, cells_te, a=.5, b=.5):
    st = {}
    for k, y in cells_tr:
        s = st.setdefault(k, [0, 0]); s[y] += 1
    ll = 0
    for k, y in cells_te:
        n0, n1 = st.get(k, [0, 0]); p = (n1 + a) / (n0 + n1 + a + b); ll += math.log(p if y else 1 - p)
    return ll
def sel(cells_tr, a=.5, b=.5):  # in-sample log-lik for selection
    st = {}
    for k, y in cells_tr:
        s = st.setdefault(k, [0, 0]); s[y] += 1
    return sum(n1 * math.log((n1 + a) / (n0 + n1 + 2 * a)) + n0 * math.log((n0 + a) / (n0 + n1 + 2 * a)) for n0, n1 in st.values())
res = {"xor2": 0, "cat2": 0, "xor2+cat1": 0, "xor2+xor2cat": 0}; nm = 0; tot = 0
LAGS = range(1, 9)
for r in q:
    if C[r["query_id"]] != "stable": continue
    s = seq[r["machine_id"]]; n = len(s); cut = int(n * .7)
    ex = [(t, int(s[t] > 0)) for t in range(8, n)]
    tr = [e for e in ex if e[0] < cut]; te = [e for e in ex if e[0] >= cut]
    def cells(fn, part): return [(fn(t), y) for t, y in part]
    xp = lambda L: (lambda t: ((s[t - L[0]] > 0) ^ (s[t - L[1]] > 0)))
    cp = lambda L: (lambda t: (cat(s[t - L[0]]), cat(s[t - L[1]])))
    bx = max(it.combinations(LAGS, 2), key=lambda L: sel(cells(xp(L), tr)))
    bc = max(it.combinations(LAGS, 2), key=lambda L: sel(cells(cp(L), tr)))
    res["xor2"] += fitll(cells(xp(bx), tr), cells(xp(bx), te))
    res["cat2"] += fitll(cells(cp(bc), tr), cells(cp(bc), te))
    bl = max(LAGS, key=lambda l: sel(cells(lambda t: (xp(bx)(t), cat(s[t - l])), tr)))
    f = lambda t: (xp(bx)(t), cat(s[t - bl])); res["xor2+cat1"] += fitll(cells(f, tr), cells(f, te))
    bl2 = max(it.combinations(LAGS, 2), key=lambda L: sel(cells(lambda t: (xp(bx)(t), xp(L)(t)), tr)))
    f = lambda t: (xp(bx)(t), xp(bl2)(t)); res["xor2+xor2cat"] += fitll(cells(f, tr), cells(f, te))
    nm += 1; tot += len(te)
print(nm, "stable machines; held-out mean log-lik per spin:")
for k, v in res.items(): print(f"  {k:14s} {v / tot:.4f}")
