"""UNBLINDING step of the pre-registered rehearsal: run only after both submissions exist. Reads the hidden truth (outside the public kit).
    python3 evaluate.py <rehearsal folder containing hidden/, out_thanos/, out_fallback/>"""
import csv, json, sys
import numpy as np

K = sys.argv[1]; COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
truth = {r["query_id"]: r for r in csv.DictReader(open(f"{K}/hidden/truth_test.csv"))}


def load(p):
    return {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(p))}


T, F = load(f"{K}/out_thanos/submission.csv"), load(f"{K}/out_fallback/submission.csv"); ids = sorted(truth); assert set(ids) == set(T) == set(F)
pt = np.array([[float(truth[i][c]) for c in COLS] for i in ids]); oh = np.zeros((len(ids), 5))
oh[np.arange(len(ids)), [(0, 1, 2, 5, 20).index(int(truth[i]["outcome"])) for i in ids]] = 1
PT, PF = np.array([T[i] for i in ids]), np.array([F[i] for i in ids])
eps = np.array([truth[i]["episode_id"] for i in ids]); cond = np.array([truth[i]["condition"] for i in ids]); typ = np.array([truth[i]["type"] for i in ids])
ue = sorted(set(eps)); EI = np.array([ue.index(e) for e in eps]); rng = np.random.default_rng(0)


def boot(d, sel):
    s = np.bincount(EI[sel], weights=d[sel], minlength=len(ue)); c = np.bincount(EI[sel], minlength=len(ue)).astype(float); k = c > 0; s, c = s[k], c[k]
    pick = rng.integers(0, len(s), (4000, len(s))); m = s[pick].sum(1) / c[pick].sum(1)
    return d[sel].mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), float((m > 0).mean())


mseT, mseF = ((PT - pt) ** 2).sum(1), ((PF - pt) ** 2).sum(1); brT, brF = ((PT - oh) ** 2).sum(1), ((PF - oh) ** 2).sum(1)
subs = [("ALL", np.ones(len(ids), bool))] + [(c, cond == c) for c in ("stable", "short_history", "drift", "shared_change")] + [(t, typ == t) for t in ("steady", "balanced", "jackpot")]
print(f"{len(ids)} test queries, {len(ue)} episodes. Thanos E2 (T) vs frozen fallback v9+shared-base (F); all x1e-4; difference = F - T (positive = Thanos better); P = share of episode-bootstrap draws with F - T > 0")
print(f"{'':15s}{'T vs truth':>11s}{'F vs truth':>11s}{'F - T [95% CI]':>25s}{'P(T better)':>12s} | {'Brier T':>8s}{'Brier F':>8s}{'F - T [95% CI]':>24s}")
for nm, sel in subs:
    m, lo, hi, p = boot(mseF - mseT, sel); b = boot(brF - brT, sel)
    print(f"{nm:15s}{mseT[sel].mean()*1e4:11.1f}{mseF[sel].mean()*1e4:11.1f}{m*1e4:+10.1f} [{lo*1e4:+.1f},{hi*1e4:+.1f}]{p:12.3f} | {brT[sel].mean()*1e4:8.0f}{brF[sel].mean()*1e4:8.0f}{b[0]*1e4:+9.1f} [{b[1]*1e4:+.1f},{b[2]*1e4:+.1f}]")
print("relative error vs truth, Thanos / fallback by condition:", {c: round(float(mseT[cond == c].mean() / mseF[cond == c].mean()), 3) for c in ("stable", "short_history", "drift", "shared_change")})
for nm in ("thanos", "fallback"): print(nm, json.load(open(f"{K}/out_{nm}/run.json")))
