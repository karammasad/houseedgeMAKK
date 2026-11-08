"""Do XOR triples (or other higher-order gates) really exist? Stable TRAIN machines (no change point; train conditions are allowed),
choose the best structure on the first half of each history by penalized log-lik, score next-tick log-loss + Brier on the second half.
Families: S = singles, P = XOR pairs, A = AND pairs (the gate's family = S+P+A), T = XOR triples. Lags 1..12."""
import csv, itertools, math, numpy as np
from common.data import load_split
PUB = "data/main"; L = 12
cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
q, seq = load_split(PUB, "train")
fams = {"S": [(l,) for l in range(1, L + 1)], "P": list(itertools.combinations(range(1, L + 1), 2)),
        "A": [("and",) + c for c in itertools.combinations(range(1, L + 1), 2)], "T": list(itertools.combinations(range(1, L + 1), 3)),
        "O": [("or",) + c for c in itertools.combinations(range(1, L + 1), 2)],
        "N": [("andnot", a, b) for a in range(1, L + 1) for b in range(1, L + 1) if a != b],
        "J": [("joint",) + c for c in itertools.combinations(range(1, L + 1), 2)]}
def cellvals(w, st):
    n = len(w); x = np.full(n, -1)
    if isinstance(st[0], str):
        a, b = st[1], st[2]
        idx = np.arange(max(a, b), n); va, vb = w[idx - a], w[idx - b]; good = (va >= 0) & (vb >= 0); va, vb = va[good], vb[good]
        x[idx[good]] = {"and": va & vb, "or": va | vb, "andnot": va & (1 - vb), "joint": 2 * va + vb}[st[0]]
        return x
    m = max(st); idx = np.arange(m, n); v = np.stack([w[idx - l] for l in st]); good = (v >= 0).all(0)
    x[idx[good]] = (v[:, good].sum(0) % 2); return x
res = {"gate": []}; picks = {}
cnt = 0
for r in q:
    if cond[r["query_id"]] != "stable": continue
    s = np.array(seq[r["machine_id"]]); w = np.where(s < 0, -1, (s > 0).astype(int)); n = len(w); h = n // 2
    if (w >= 0).sum() < 150: continue
    cnt += 1
    best = {}
    for f, sts in fams.items():
        for st in sts:
            x = cellvals(w, st); tr = (np.arange(n) < h) & (x >= 0) & (w >= 0)
            ll = 0.0; k = []
            for c in ((0, 1, 2, 3) if st[0] == "joint" else (0, 1)):
                y = w[tr & (x == c)]; a = y.sum() + 1; b = len(y) - y.sum() + 1
                ll += math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)     # Beta(1,1) marginal likelihood
                k.append((a, b))
            best.setdefault(f, []).append((ll, st, k))
    for f in best: best[f].sort(key=lambda z: -z[0])
    def heldout(choice):
        ll, st, k = choice; x = cellvals(w, st); te = (np.arange(n) >= h) & (x >= 0) & (w >= 0)
        p = np.array([k[c][0] / (k[c][0] + k[c][1]) for c in x[te]]); y = w[te]
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))), np.mean((p - y) ** 2), te.sum()
    G = max(best["S"][0], best["P"][0], best["A"][0], key=lambda z: z[0])
    for alt in ("O", "N", "J", "T"):
        fam_alt = best[alt][0]
        pen = math.log(len(fams[alt]) / 79.0) + (2.0 if alt == "J" else 0.0)
        pick = fam_alt if fam_alt[0] - pen > G[0] + 2.0 else G
        res.setdefault("gate+" + alt, []).append(heldout(pick)); res.setdefault(alt + "_only", []).append(heldout(fam_alt))
        picks[alt] = picks.get(alt, 0) + (pick is fam_alt)
    res["gate"].append(heldout(G))
print("stable train machines", cnt, "alt family picked over gate family:", picks)
for f, v in res.items():
    if not v: continue
    v = np.array(v); print(f"{f:9s} held-out logloss {v[:, 0].mean():.4f}  brier {v[:, 1].mean():.4f}")
base = np.array(res["gate"])
for alt in ("O", "N", "J", "T"):
    d = np.array(res["gate+" + alt])[:, 1] - base[:, 1]; print("brier gate+%s - gate: %+.5f ± %.5f" % (alt, d.mean(), d.std() / math.sqrt(len(d))))
