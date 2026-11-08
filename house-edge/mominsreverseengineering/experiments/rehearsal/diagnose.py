"""POST-HOC diagnosis (after unblinding; labelled as such, not used to alter either model): where does Thanos E2 lose to the fallback on the rehearsal kit?
Uses the deterministic regeneration's hidden structure / flip-time labels (public kit files verified byte-identical)."""
import csv, sys
import numpy as np
K = sys.argv[1]; K2 = sys.argv[2]; COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
truth = {r["query_id"]: r for r in csv.DictReader(open(f"{K}/hidden/truth_test.csv"))}; meta = {r["query_id"]: r for r in csv.DictReader(open(f"{K2}/hidden/meta_test.csv"))}
load = lambda p: {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(p))}
T, F = load(f"{K}/out_thanos/submission.csv"), load(f"{K}/out_fallback/submission.csv"); ids = sorted(truth)
pt = np.array([[float(truth[i][c]) for c in COLS] for i in ids]); eT = ((np.array([T[i] for i in ids]) - pt) ** 2).sum(1) * 1e4; eF = ((np.array([F[i] for i in ids]) - pt) ** 2).sum(1) * 1e4
ep = np.array([truth[i]["episode_id"] for i in ids]); ue = sorted(set(ep)); EI = np.array([ue.index(e) for e in ep]); rng = np.random.default_rng(0)
cond = np.array([truth[i]["condition"] for i in ids]); typ = np.array([truth[i]["type"] for i in ids]); gate = np.array(["none" if meta[i]["gate"] == "None" else ("single" if len(eval(meta[i]["gate"])) == 1 else "pair") for i in ids])
tau = np.array([float(meta[i]["tau"]) / float(meta[i]["cutoff"]) if meta[i]["tau"] else np.nan for i in ids]); pw = 1 - pt[:, 0]
def row(nm, sel):
    d = (eF - eT); s = np.bincount(EI[sel], weights=d[sel], minlength=len(ue)); c = np.bincount(EI[sel], minlength=len(ue)).astype(float); k = c > 0; s, c = s[k], c[k]
    m = np.sort(s[rng.integers(0, len(s), (3000, len(s)))].sum(1) / np.maximum(c.sum(), 1)) if False else None
    pick = rng.integers(0, len(s), (3000, len(s))); m = s[pick].sum(1) / c[pick].sum(1)
    print(f"{nm:44s} n={int(sel.sum()):5d}  T {eT[sel].mean():7.1f}  F {eF[sel].mean():7.1f}  F-T {d[sel].mean():+8.1f} [{np.percentile(m,2.5):+.1f},{np.percentile(m,97.5):+.1f}]")
print("x1e-4 squared distance to the TRUE vector; F - T > 0 means Thanos better\n-- by true gate kind")
for g in ("pair", "single", "none"): row(f"gate = {g}", gate == g)
print("-- steady type split by gate kind and by extreme true win probability")
for g in ("pair", "single", "none"): row(f"steady, gate = {g}", (typ == "steady") & (gate == g))
row("steady, true P(win) > 0.97", (typ == "steady") & (pw > .97)); row("steady, true P(win) <= 0.97", (typ == "steady") & (pw <= .97))
print("-- drift condition by flip position tau/cutoff (window assumption: first 45% flip-free)")
dr = cond == "drift"
for lo, hi in ((0, .45), (.45, .55), (.55, .65), (.65, .75), (.75, 1.0)): row(f"drift, tau/L in [{lo},{hi})", dr & (tau >= lo) & (tau < hi))
print("-- all conditions excluding no-gate machines (the structure family Thanos keeps)"); row("gated only, ALL", gate != "none")
for c in ("stable", "short_history", "drift", "shared_change"): row(f"gated only, {c}", (gate != "none") & (cond == c))
print("-- no-gate machines by type"); 
for t in ("steady", "balanced", "jackpot"): row(f"no gate, {t}", (gate == "none") & (typ == t))
