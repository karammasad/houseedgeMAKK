import pickle, glob
import numpy as np
D = [pickle.load(open(f, "rb")) for f in sorted(glob.glob("real_cv_fold[0-9].pkl"))]; meta = sum([d["meta"] for d in D], []); P = {v: np.concatenate([d["preds"][v] for d in D]) for v in ("T0", "T1")}
y = np.array([m["y"] for m in meta]); oh = np.zeros((len(y), 5)); oh[np.arange(len(y)), y] = 1; B = {v: ((P[v] - oh) ** 2).sum(1) for v in P}
ep = np.array([m["ep"] for m in meta]); ue = sorted(set(ep)); EI = np.array([ue.index(e) for e in ep]); rng = np.random.default_rng(0)
def row(nm, sel):
    d = B["T1"] - B["T0"]; s = np.bincount(EI[sel], weights=d[sel], minlength=len(ue)); c = np.bincount(EI[sel], minlength=len(ue)).astype(float); k = c > 0; s, c = s[k], c[k]
    pick = rng.integers(0, len(s), (4000, len(s))); m = s[pick].sum(1) / c[pick].sum(1); print(f"{nm:20s} n={int(sel.sum()):5d}  T1 - T0 {d[sel].mean()*1e4:+7.2f} [{np.percentile(m,2.5)*1e4:+.2f},{np.percentile(m,97.5)*1e4:+.2f}] x1e-4 (negative = adaptation better)"); return np.percentile(m, 97.5) * 1e4
print(f"folds {[d['meta'][0]['fold'] for d in D]}, {len(y)} held-out rows, {len(ue)} episodes; absolute Brier x1e-4 T0 {B['T0'].mean()*1e4:.1f}  T1 {B['T1'].mean()*1e4:.1f}")
ub = row("ALL", np.ones(len(y), bool))
for t in ("steady", "balanced", "jackpot"): row(t, np.array([m["type"] == t for m in meta]))
for c in ("stable", "short_history", "drift", "shared_change"): row(c, np.array([m["cond"] == c for m in meta]))
print("learned no-gate prior mass per fold and mu_n/ls_n per type:"); [print("  fold", d["meta"][0]["fold"], d["info"]) for d in D]
print(f"PRE-REGISTERED CHECK: upper 95% bound of T1 - T0 = {ub:+.2f} <= +3.0 : {ub <= 3.0}")
