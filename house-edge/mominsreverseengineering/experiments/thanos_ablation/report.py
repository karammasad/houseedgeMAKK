import pickle, glob
import numpy as np
from collections import defaultdict
D = [pickle.load(open(f, "rb")) for f in sorted(glob.glob("fold[0-9].pkl"))]
meta = sum([d["meta"] for d in D], []); V = list(D[0]["preds"]); P = {v: np.concatenate([d["preds"][v] for d in D]) for v in V}
y = np.array([m["y"] for m in meta]); oh = np.zeros((len(y), 5)); oh[np.arange(len(y)), y] = 1
B = {v: ((P[v] - oh) ** 2).sum(1) for v in V}; ep = np.array([m["ep"] for m in meta]); ue = sorted(set(ep)); EI = np.array([ue.index(e) for e in ep]); rng = np.random.default_rng(0)
def pair(a, b, sel):
    d = B[a] - B[b]; s = np.bincount(EI[sel], weights=d[sel], minlength=len(ue)); c = np.bincount(EI[sel], minlength=len(ue)).astype(float); k = c > 0; s, c = s[k], c[k]
    pick = rng.integers(0, len(s), (4000, len(s))); m = s[pick].sum(1) / c[pick].sum(1); return d[sel].mean() * 1e4, np.percentile(m, 2.5) * 1e4, np.percentile(m, 97.5) * 1e4, float((m < 0).mean())
cond = np.array([m["cond"] for m in meta]); typ = np.array([m["type"] for m in meta]); lag0 = np.array([m["lag"] == 0 for m in meta])
subs = [("ALL", np.ones(len(y), bool))] + [(c, cond == c) for c in ("stable", "drift", "shared_change", "short_history")] + [(t, typ == t) for t in ("steady", "balanced", "jackpot")] + [("real cutoff only (lag 0)", lag0)]
print(f"{len(y)} held-out rows, {len(ue)} episodes, 4 episode-grouped folds (hyperparameters from the other folds' train histories). 5-class Brier x1e-4 (lower is better)")
print("absolute Brier x1e-4:", {v: round(float(B[v].mean()) * 1e4, 1) for v in V})
for a, nm in (("nodir", "direction tie REMOVED (independent d,e)"), ("flat-pos", "flip POSITION prior flat"), ("flat-flip", "flip prior flat AND P(no flip)=.5"), ("common", "tick set common instead of own (E2 question)")):
    print(f"\n{nm}: variant - full  (positive = variant worse; P = share of bootstrap draws where the variant is worse)")
    for sn, sel in subs:
        m, lo, hi, p = pair(a, "full", sel); print(f"   {sn:26s} n={int(sel.sum()):5d} {m:+8.2f} [{lo:+.2f},{hi:+.2f}]  P(worse)={1-p:.2f}")
I = D[0]["info"]
for d in D: print("fold", d["meta"][0]["fold"], "P(d,e) learned", np.round(np.array(d["info"]["Pde_full"]).ravel(), 3), "rho", {k: round(v, 3) for k, v in d["info"]["rho_full"].items()})
