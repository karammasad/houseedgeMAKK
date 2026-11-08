import pickle, glob
import numpy as np
D = [pickle.load(open(f, "rb")) for f in sorted(glob.glob("thr_fold[0-9].pkl"))]
meta = sum([d["meta"] for d in D], []); V = list(D[0]["preds"]); P = {v: np.concatenate([d["preds"][v] for d in D]) for v in V}
y = np.array([m["y"] for m in meta]); oh = np.zeros((len(y), 5)); oh[np.arange(len(y)), y] = 1; B = {v: ((P[v] - oh) ** 2).sum(1) for v in V}
ep = np.array([m["ep"] for m in meta]); ue = sorted(set(ep)); EI = np.array([ue.index(e) for e in ep]); rng = np.random.default_rng(0)
def pair(a, b, sel):
    d = B[a] - B[b]; s = np.bincount(EI[sel], weights=d[sel], minlength=len(ue)); c = np.bincount(EI[sel], minlength=len(ue)).astype(float); k = c > 0; s, c = s[k], c[k]
    pick = rng.integers(0, len(s), (4000, len(s))); m = s[pick].sum(1) / c[pick].sum(1); return d[sel].mean() * 1e4, np.percentile(m, 2.5) * 1e4, np.percentile(m, 97.5) * 1e4
cond = np.array([m["cond"] for m in meta]); typ = np.array([m["type"] for m in meta]); sh = np.array([m["short"] for m in meta])
subs = [("ALL", np.ones(len(y), bool))] + [(c, cond == c) for c in ("stable", "drift", "shared_change", "short_history")] + [(t, typ == t) for t in ("steady", "balanced", "jackpot")]
print(f"folds {[d['meta'][0]['fold'] for d in D]}: {len(y)} held-out rows, {len(ue)} episodes; 5-class Brier x1e-4 (variant - 1% rule; negative = wider support better)")
print("absolute:", {v: round(float(B[v].mean()) * 1e4, 1) for v in V})
for a in V[1:]:
    print(f"\n{a}")
    for sn, sel in subs: m, lo, hi = pair(a, "thr1%", sel); print(f"   {sn:14s} n={int(sel.sum()):5d} {m:+7.2f} [{lo:+.2f},{hi:+.2f}]")
print("\nsupport size K and prediction seconds per fold (768 rows):")
for d in D: print("  fold", d["meta"][0]["fold"], {k: tuple(round(x, 1) if isinstance(x, float) else x for x in v) for k, v in d["timing"].items()})
a, b = np.concatenate([d["preds"]["thr1%"] for d in D]), np.concatenate([d["preds"]["thr0.1%"] for d in D]); dd = np.abs(a - b).max(1)
print(f"\nprediction change from widening to 0.1%: mean |dp| {np.abs(a-b).mean():.5f}, rows with max |dp| > 0.02: {int((dd>.02).sum())} of {len(dd)}, largest {dd.max():.3f}")
