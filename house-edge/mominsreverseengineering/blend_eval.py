"""Pre-registered 50/50 blend of v8 with adi.cook, evaluated on the TUNING half only (the audit half is never read here).

Identity used (exact, per query): |w a + (1-w) c - y|^2 = w |a - y|^2 + (1-w) |c - y|^2 - w (1-w) |a - c|^2.
So a blend beats model A only if the models disagree (|a - c|^2) by more than their error gap. The decomposition is printed per condition.
The weight is FIXED at 0.5 before looking (no tuning); other weights are printed for information only and are never selected.
Run: python3 blend_eval.py --a experiments/pred_base.pkl --b experiments/pred_cook.pkl [--w 0.5]
"""
import argparse, csv, os, pickle, random
from collections import defaultdict
from common.data import public_dir
from common.evaluate import folds, points, COLS, CONDITIONS


def blend(A, B, w):
    return {q: [w * x + (1 - w) * y for x, y in zip(A[q], B[q])] for q in A}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--data", default="../data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub)                       # TUNING half only
    A = pickle.load(open(a.a, "rb")); B = pickle.load(open(a.b, "rb")); W = blend(A, B, a.w)
    meta = __import__("json").load(open(os.path.join(pub, "metadata.json"))); mp = float(meta.get("max_points", 60))
    sa, ra = points(pub, A, tune, mp); sb, rb = points(pub, B, tune, mp); sw, rw = points(pub, W, tune, mp)
    print(f"TUNING official (/{mp:g}):  A (v8) {sa:.4f}   B (adi.cook) {sb:.4f}   blend w={a.w} {sw:.4f}   blend - A = {sw - sa:+.4f}")
    print(f"{'condition':14s} {'A ratio':>9s} {'B ratio':>9s} {'blend':>9s}  blend vs A")
    for c in CONDITIONS:
        print(f"{c:14s} {ra[c]:9.5f} {rb[c]:9.5f} {rw[c]:9.5f}  {rw[c] / ra[c] - 1:+.1%}")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}; ref = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("reference.csv")}
    sq = lambda p, t: sum((x / sum(p) - y) ** 2 for x, y in zip(p, t))
    E = {k: defaultdict(lambda: defaultdict(list)) for k in "ABWD"}; R = defaultdict(lambda: defaultdict(list))
    for t in rd("targets.csv"):
        if t["episode_id"] not in tune:
            continue
        q = t["query_id"]; tr = [float(t[c]) for c in COLS]; c = cond[q]; ep = t["episode_id"]
        E["A"][c][ep].append(sq(A[q], tr)); E["B"][c][ep].append(sq(B[q], tr)); E["W"][c][ep].append(sq(W[q], tr))
        E["D"][c][ep].append(sum((x - y) ** 2 for x, y in zip(A[q], B[q]))); R[c][ep].append(sum((x - y) ** 2 for x, y in zip(ref[q], tr)))
    m = lambda d: {c: {e: sum(v) / len(v) for e, v in d[c].items()} for c in CONDITIONS}
    EA, EB, EW, ED, RR = m(E["A"]), m(E["B"]), m(E["W"]), m(E["D"]), m(R)
    print("\nIDENTITY CHECK and decomposition in reference-normalised units (per condition; mean over episodes):")
    print(f"{'condition':14s} {'E_A':>9s} {'E_B':>9s} {'E_blend':>9s} {'w E_A+(1-w)E_B - w(1-w)D':>26s}  {'disagreement D':>14s}  {'gap E_B-E_A':>11s}")
    for c in CONDITIONS:
        n = lambda d: sum(d[c].values()) / len(d[c]); r = sum(RR[c].values()) / len(RR[c])
        pred = (a.w * n(EA) + (1 - a.w) * n(EB) - a.w * (1 - a.w) * n(ED)) / r
        print(f"{c:14s} {n(EA) / r:9.5f} {n(EB) / r:9.5f} {n(EW) / r:9.5f} {pred:26.5f}  {n(ED) / r:14.5f}  {(n(EB) - n(EA)) / r:+11.5f}")
    eps = {c: sorted(RR[c]) for c in CONDITIONS}; rng = random.Random(12345); wins = 0; g = []
    Rof = lambda Ex, d: sum(sum(Ex[c][e] for e in d[c]) / sum(RR[c][e] for e in d[c]) for c in CONDITIONS) / 4
    for _ in range(4000):
        d = {c: [rng.choice(eps[c]) for _ in eps[c]] for c in CONDITIONS}
        x = mp / (1 + Rof(EW, d)) - mp / (1 + Rof(EA, d)); g.append(x); wins += x > 0
    g.sort(); print(f"\npaired episode bootstrap (48 tuning episodes), blend - A: {sw - sa:+.4f} points, 95% CI [{g[100]:+.4f}, {g[3900]:+.4f}], P(blend better) = {wins / 4000:.3f}")
    print("episodes where the blend beats A: " + ", ".join(f"{c} {sum(1 for e in eps[c] if EW[c][e] < EA[c][e])}/{len(eps[c])}" for c in CONDITIONS))
    print("\nFOR INFORMATION ONLY (never selected): tuning score by blend weight on A: " + ", ".join(
        f"w={x:.2f} {points(pub, blend(A, B, x), tune, mp)[0]:.4f}" for x in (0.0, 0.25, 0.5, 0.75, 1.0)))


if __name__ == "__main__":
    main()
