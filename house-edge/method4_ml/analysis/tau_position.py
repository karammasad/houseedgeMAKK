"""Where do changes happen? (train: posterior over flip time; validation: error vs flip recency)."""
import csv, math, statistics as st
from collections import defaultdict
from common.data import load_split
from method4_ml.model import Model
from adi import gate as G
PUB = "data/main"
gm = Model(blend=0).fit(PUB).gm
def tau_post(r, s):
    cut = r["cutoff"]; cfg = gm._mcfg(s, r_type=r["machine_type"])
    taus, lprior = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(s, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r)); sp = gm._sp(s)
    J = defaultdict(list)
    for si, rows in enumerate(tb):
        if sp[si] <= 0: continue
        for ti, (ll, pr) in enumerate(rows): J[ti].append(ll + math.log(sp[si]) + lprior[ti])
    L = {ti: G.logsumexp(v) for ti, v in J.items()}; z = G.logsumexp(list(L.values()))
    return {(taus[ti] if ti < len(taus) else None): math.exp(v - z) for ti, v in L.items()}, cut, taus
for split in ("train",):
    q, seq = load_split(PUB, split)
    C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/{split}/conditions.csv"))}
    hist = defaultdict(lambda: defaultdict(float)); n = defaultdict(int)
    for r in q:
        s = seq[r["machine_id"]]
        if sum(x < 0 for x in s) > len(s) / 2: continue
        post, cut, taus = tau_post(r, s); c = C[r["query_id"]]; n[c] += 1
        for t, p in post.items():
            key = "none" if t is None else min((cut - t) // 20, 9) * 20
            hist[c][key] += p
    for c in ("stable", "drift", "shared_change"):
        h = hist[c]; tot = n[c]
        print(f"{split} {c:13s} P(no change) {h['none'] / tot:.2f} | P(change at cut-age in [a,a+20)) by a: " +
              " ".join(f"{a}:{h[a] / tot:.2f}" for a in range(0, 200, 20)))
# validation: error vs recency for flipped queries
T = {r["query_id"]: 1 - float(r["p_0"]) for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
Cv = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/validation/conditions.csv"))}
q, seq = load_split(PUB, "validation"); agg = defaultdict(lambda: [0, 0.0])
for r in q:
    s = seq[r["machine_id"]]
    if sum(x < 0 for x in s) > len(s) / 2: continue
    post, cut, taus = tau_post(r, s)
    pc = 1 - post.get(None, 0)
    if pc < .5: continue
    age = sum(p * (cut - t) for t, p in post.items() if t is not None) / pc
    pw = G.win_probability(r, s, gm.ST, gm._sp(s), cellprior=gm._cp(r), **gm._mcfg(s, r_type=r["machine_type"]))
    b = min(int(age // 20) * 20, 140); x = agg[(Cv[r["query_id"]], b)]; x[0] += 1; x[1] += (pw - T[r["query_id"]]) ** 2
print("\nvalidation, machines with P(change) > .5: P(win) MSE by expected change age (cutoff - tau)")
for k in sorted(agg): print(f"   {k[0]:13s} age {k[1]:3d}-{k[1] + 19:3d}: n={agg[k][0]:3d} mse {agg[k][1] / agg[k][0]:.4f}")
