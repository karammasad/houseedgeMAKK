"""Do machines of a shared_change episode flip at the SAME tick? (train, conditions used for analysis only.)
Per machine: posterior-mean flip time (adi gate). Within-episode sd vs between-episode sd of tau, shared vs drift episodes."""
import csv, math, statistics as st
from collections import defaultdict
from common.data import load_split
from method4_ml.model import Model
from adi import gate as G
PUB = "data/main"; gm = Model(blend=0).fit(PUB).gm
q, seq = load_split(PUB, "train")
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/train/conditions.csv"))}
tau = defaultdict(list)
for r in q:
    s = seq[r["machine_id"]]; c = C[r["query_id"]]
    if c not in ("shared_change", "drift") or sum(x < 0 for x in s) > len(s) / 2: continue
    cut = r["cutoff"]; cfg = gm._mcfg(s, r_type=r["machine_type"]); taus, lp = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(s, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r)); sp = gm._sp(s)
    J = [[tb[i][ti][0] + math.log(sp[i]) + lp[ti] for ti in range(len(taus))] for i in range(len(tb)) if sp[i] > 0]
    L = [G.logsumexp([row[ti] for row in J]) for ti in range(len(taus))]; z = G.logsumexp(L)
    w = [math.exp(x - z) for x in L]; tau[(c, r["episode_id"])].append(sum(a * t for a, t in zip(w, taus)) / sum(w))
for c in ("shared_change", "drift"):
    eps = {e: v for (cc, e), v in tau.items() if cc == c and len(v) >= 4}
    within = [st.pstdev(v) for v in eps.values()]; means = [st.mean(v) for v in eps.values()]
    allv = [x for v in eps.values() for x in v]
    print(f"{c:13s} episodes {len(eps)}: median within-episode sd of tau {st.median(within):5.1f} | sd of episode means {st.pstdev(means):5.1f} | overall sd {st.pstdev(allv):5.1f}")
