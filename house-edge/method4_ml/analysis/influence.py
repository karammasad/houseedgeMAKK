"""Experiment 5 (2026-11-08): where does Momin's shared-base gain over v8 come from? Influence analysis on full validation.

Per-episode contribution to the change in R (official metric): d_e = (err_new_e - err_old_e) / (n_c * B_c * 4), where n_c = episodes in the
episode's condition and B_c = the frozen reference's condition error; points change ~ -60 / (1 + R)^2 * sum(d_e) (first order; the exact
change is printed too). Query-level contributions split d_e by machine type. Leave-one-out = exact points of both models without that
episode. Diagnosis only (the audit half is already consumed; nothing here selects a model).
  python -m method4_ml.analysis.influence <momin_val.pkl>
"""
import os, pickle, sys
from collections import defaultdict
import numpy as np
from common.evaluate import points, _rows, COLS, CONDITIONS
from common.data import load_split

PUB = "data/main"


def main():
    new = pickle.load(open(sys.argv[1], "rb"))
    old = pickle.load(open(sys.argv[2], "rb")) if len(sys.argv) > 2 else \
        pickle.load(open(os.path.join(os.path.dirname(__file__), "..", "cache", "gated_pool_preds_main.pkl"), "rb"))["validation"][0]
    T = _rows(os.path.join(PUB, "validation", "targets.csv")); C = {r["query_id"]: r["condition"] for r in _rows(os.path.join(PUB, "validation", "conditions.csv"))}
    REF = {r["query_id"]: [float(r[c]) for c in COLS] for r in _rows(os.path.join(PUB, "validation", "reference.csv"))}
    q, seq = load_split(PUB, "validation"); TY = {r["query_id"]: r["machine_type"] for r in q}
    se = lambda p, t: sum((a - b) ** 2 for a, b in zip(p, t))
    epq = defaultdict(list); cond = {}
    for t in T:
        tr = [float(t[c]) for c in COLS]; k = t["query_id"]
        epq[t["episode_id"]].append((k, se(old[k], tr), se(new[k], tr), se(REF[k], tr))); cond[t["episode_id"]] = C[k]
    B = {c: np.mean([np.mean([x[3] for x in epq[e]]) for e in epq if cond[e] == c]) for c in CONDITIONS}
    n = {c: sum(cond[e] == c for e in epq) for c in CONDITIONS}
    p_old, p_new = points(PUB, old)[0], points(PUB, new)[0]; R = 60 / p_old - 1; scale = -60 / (1 + R) ** 2
    d = {e: sum(x[2] - x[1] for x in v) / len(v) / (n[cond[e]] * B[cond[e]] * 4) for e, v in epq.items()}
    pts = {e: scale * v for e, v in d.items()}                   # first-order points contribution of each episode (+ = Momin better)
    eps = sorted(pts, key=lambda e: -pts[e]); arr = np.array([pts[e] for e in eps])
    print(f"v8 {p_old:.4f}  momin {p_new:.4f}  exact change {p_new - p_old:+.4f}  first-order sum {arr.sum():+.4f}")
    print(f"episode contributions (points): mean {arr.mean():+.5f}  MEDIAN {np.median(arr):+.5f}  improved {np.sum(arr > 0)}/{len(arr)}")
    rng = np.random.default_rng(0); by = defaultdict(list)
    for e in eps: by[cond[e]].append(pts[e])
    bs = [sum(np.mean(rng.choice(v, len(v))) * len(v) for v in by.values()) for _ in range(5000)]
    print(f"bootstrap (within condition) 95% CI of the change: [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], P(>0) {np.mean(np.array(bs) > 0):.3f}")
    for c in CONDITIONS:
        v = np.array(by[c]); print(f"  {c:14s} sum {v.sum():+.4f}  median {np.median(v):+.5f}  improved {np.sum(v > 0)}/{len(v)}")
    ty = defaultdict(float)
    for e, v in epq.items():
        for k, o, nw, _ in v:
            ty[TY[k]] += scale * (nw - o) / len(v) / (n[cond[e]] * B[cond[e]] * 4)
    print("by machine type: " + "  ".join(f"{t} {x:+.4f}" for t, x in sorted(ty.items())))
    tot = arr.sum(); print("top 5 episodes: " + ", ".join(f"{e[-6:]} ({cond[e]}) {pts[e]:+.4f}" for e in eps[:5]) + f"  -> {sum(arr[:5]) / tot:.0%} of the gain")
    print("bottom 3: " + ", ".join(f"{e[-6:]} ({cond[e]}) {pts[e]:+.4f}" for e in eps[-3:]))
    for k in (5, 10):
        print(f"without the top {k}: {tot - arr[:k].sum():+.4f}")
    def pts_without(pred, drop):
        keep = {e for e in epq if e not in drop}; return points(PUB, pred, keep)[0]
    print("leave-one-out (exact, each top-5 episode removed): " + ", ".join(f"{pts_without(new, {e}) - pts_without(old, {e}):+.4f}" for e in eps[:5]))
    # jackpot share of each episode vs its contribution
    js = np.array([np.mean([TY[k] == "jackpot" for k, *_ in epq[e]]) for e in eps]); hi = js >= np.median(js)
    print(f"jackpot-heavy half of episodes: {arr[hi].sum():+.4f}  other half: {arr[~hi].sum():+.4f}  (corr share vs contribution {np.corrcoef(js, arr)[0, 1]:+.2f})")


if __name__ == "__main__":
    main()
