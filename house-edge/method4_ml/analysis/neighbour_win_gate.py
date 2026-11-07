"""Does a NEIGHBOUR machine's past win (same episode, lag l) add information to a machine's own win gate?
STABLE long train machines; own structure (best XOR pair) + best (neighbour, lag) chosen on the first 70% of ticks,
scored on the last 30% (per-spin log-lik). Compared: own gate only / own x neighbour split / own XOR neighbour."""
import csv, itertools as it, math
from collections import defaultdict
from common.data import load_split
PUB = "data/main"
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
eps = defaultdict(list)
for r in q: eps[r["episode_id"]].append(r["machine_id"])
def fit_score(tr, te, a=.5):
    st = {}
    for k, y in tr: s = st.setdefault(k, [0, 0]); s[y] += 1
    ll = 0
    for k, y in te:
        n0, n1 = st.get(k, [0, 0]); p = (n1 + a) / (n0 + n1 + 2 * a); ll += math.log(p if y else 1 - p)
    return ll
def sel(tr, a=.5):
    st = {}
    for k, y in tr: s = st.setdefault(k, [0, 0]); s[y] += 1
    return sum(n1 * math.log((n1 + a) / (n0 + n1 + 2 * a)) + n0 * math.log((n0 + a) / (n0 + n1 + 2 * a)) for n0, n1 in st.values())
res = defaultdict(float); tot = 0; nm = 0
for r in q:
    if C[r["query_id"]] != "stable": continue
    mid = r["machine_id"]; s = seq[mid]; n = len(s); cut = int(n * .7)
    own = lambda t, L: (s[t - L[0]] > 0) ^ (s[t - L[1]] > 0)
    T = range(8, n); tr = [t for t in T if t < cut]; te = [t for t in T if t >= cut]
    bx = max(it.combinations(range(1, 9), 2), key=lambda L: sel([(own(t, L), int(s[t] > 0)) for t in tr]))
    nb = [m for m in eps[r["episode_id"]] if m != mid]
    cand = [(m, l) for m in nb for l in range(1, 9)]
    def mk(f, ts): return [(f(t), int(s[t] > 0)) for t in ts]
    f_own = lambda t: own(t, bx)
    res["own"] += fit_score(mk(f_own, tr), mk(f_own, te))
    best = max(cand, key=lambda c: sel(mk(lambda t: (f_own(t), seq[c[0]][t - c[1]] > 0), tr)))
    f_split = lambda t: (f_own(t), seq[best[0]][t - best[1]] > 0); res["own x neighbour"] += fit_score(mk(f_split, tr), mk(f_split, te))
    best2 = max(cand, key=lambda c: sel(mk(lambda t: f_own(t) ^ (seq[c[0]][t - c[1]] > 0), tr)))
    f_x = lambda t: f_own(t) ^ (seq[best2[0]][t - best2[1]] > 0); res["own XOR neighbour"] += fit_score(mk(f_x, tr), mk(f_x, te))
    # control: same procedure using the machine's own extra lag (known no-gain) for calibration of selection bias
    own_c = [(m, l) for m in [mid] for l in range(1, 9)]
    best3 = max(own_c, key=lambda c: sel(mk(lambda t: (f_own(t), seq[c[0]][t - c[1]] > 0), tr)))
    f_c = lambda t: (f_own(t), seq[best3[0]][t - best3[1]] > 0); res["own x own-lag (control)"] += fit_score(mk(f_c, tr), mk(f_c, te))
    tot += len(te); nm += 1
print(nm, "stable machines; held-out mean log-lik per spin")
for k, v in res.items(): print(f"  {k:26s} {v / tot:.4f}")
