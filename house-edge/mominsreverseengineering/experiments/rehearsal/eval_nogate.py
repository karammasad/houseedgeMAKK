"""Evaluation of PREREG_NOGATE.md. python3 eval_nogate.py <rehearsal2 folder>   (Test A; Test B is eval_real_cv.py)"""
import csv, json, sys
import numpy as np
K = sys.argv[1]; COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
truth = {r["query_id"]: r for r in csv.DictReader(open(f"{K}/hidden/truth_test.csv"))}; meta = {r["query_id"]: r for r in csv.DictReader(open(f"{K}/hidden/meta_test.csv"))}
load = lambda p: {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(p))}
A, B = load(f"{K}/out_T0/submission.csv"), load(f"{K}/out_T1/submission.csv"); ids = sorted(truth); assert set(ids) == set(A) == set(B)
pt = np.array([[float(truth[i][c]) for c in COLS] for i in ids]); e0 = ((np.array([A[i] for i in ids]) - pt) ** 2).sum(1) * 1e4; e1 = ((np.array([B[i] for i in ids]) - pt) ** 2).sum(1) * 1e4
ep = np.array([truth[i]["episode_id"] for i in ids]); ue = sorted(set(ep)); EI = np.array([ue.index(e) for e in ep]); rng = np.random.default_rng(0)
cond = np.array([truth[i]["condition"] for i in ids]); typ = np.array([truth[i]["type"] for i in ids]); gate = np.array(["none" if meta[i]["gate"] == "None" else "gated" for i in ids])
def row(nm, sel):
    d = e0 - e1; s = np.bincount(EI[sel], weights=d[sel], minlength=len(ue)); c = np.bincount(EI[sel], minlength=len(ue)).astype(float); k = c > 0; s, c = s[k], c[k]
    m = s[rng.integers(0, len(s), (4000, len(s)))].sum(1) / c.sum() if False else None; pick = rng.integers(0, len(s), (4000, len(s))); m = s[pick].sum(1) / c[pick].sum(1)
    print(f"{nm:30s} n={int(sel.sum()):5d}  T0 {e0[sel].mean():7.1f}  T1 {e1[sel].mean():7.1f}  T0-T1 {d[sel].mean():+8.2f} [{np.percentile(m,2.5):+.2f},{np.percentile(m,97.5):+.2f}]  P(T1 better) {(m>0).mean():.3f}")
    return d[sel].mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
print("x1e-4 squared distance to the TRUE vector; T0 - T1 > 0 means the adaptation helps")
o = row("ALL", np.ones(len(ids), bool)); g = row("machines with a gate", gate == "gated"); n = row("no-gate machines", gate == "none")
for t in ("steady", "balanced", "jackpot"): row(f"no gate, {t}", (gate == "none") & (typ == t))
for c in ("stable", "short_history", "drift", "shared_change"): row(f"gated, {c}", (gate == "gated") & (cond == c))
r0, r1 = json.load(open(f"{K}/out_T0/run.json")), json.load(open(f"{K}/out_T1/run.json")); print("T0", r0); print("T1", r1)
print(f"\nPRE-REGISTERED CHECKS: (i) overall improvement CI excludes 0: {o[1] > 0}   (ii) gated machines worsening upper bound <= 1.0: worsening upper bound = {-g[1]:+.2f} -> {-g[1] <= 1.0}   (iii) fit overhead {100*(r1['fit_s']/r0['fit_s']-1):+.0f}% <= 25%: {r1['fit_s']/r0['fit_s'] <= 1.25}")
