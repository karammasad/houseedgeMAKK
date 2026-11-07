"""Experiment 11 evaluation: learned flip-time prior vs adi's flat prior (tuning half; audit only via keepcheck).

  python -m method2_sizes.flip_eval MODEL.pkl --data data/main
MODEL.pkl = pickled fitted FINAL model (so the 250 s fit is not repeated). Population parameters (flip prior) are
fitted on TRAIN histories only; the win gate and sizes are unchanged.
"""
import argparse, csv, math, os, pickle, sys, time
from common.data import load_split, public_dir
from common.evaluate import folds, points, COLS, CONDITIONS
from method2_sizes import flipprior as F
from adi import gate as G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkl"); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    sys.setrecursionlimit(10000)
    pub = public_dir(a.data); tune, _ = folds(pub); m = pickle.load(open(a.pkl, "rb")); win, sizes = m.win, m.sizes
    vq, vs = load_split(pub, "validation")
    # 1. exactness: flat prior reproduces adi's win probability
    worst = 0.0
    for r in vq[:25]:
        s = vs[r["machine_id"]]; P = F.pieces(win, r, s); nt = len(P["taus"]); idx, cnt = F._bins(P["taus"], P["cutoff"] if "cutoff" in P else P["cut"])
        cfg = win._mcfg(s, r_type=r["machine_type"]); pc = cfg["p_change"]
        pi = [pc * cnt[b] / nt for b in range(F.B)]
        ref = G.win_probability(r, s, win.ST, win._sp(s), cellprior=win._cp(r), **cfg)
        worst = max(worst, abs(F.posterior(P, pi, 1 - pc)[0] - ref))
    print(f"exactness: max |flat-prior pwin - adi.win_probability| over 25 queries = {worst:.2e}")
    t0 = time.time(); tq, ts = load_split(pub, "train"); fp = F.FlipPrior().fit(win, tq, ts); tfit = time.time() - t0
    print(f"flip prior fit on train: {tfit:.0f}s ->", fp.summary())
    pi = fp.pi[False]; print("  long-log histogram of relative flip position (bins of 0.025, mass>1%):",
                             [(round((b + .5) / F.B, 3), round(x, 3)) for b, x in enumerate(pi) if x > .01])
    t0 = time.time(); base, new = {}, {}
    for r in vq:
        s = vs[r["machine_id"]]; sz = sizes.predict(r, s)
        p0 = G.win_probability(r, s, win.ST, win._sp(s), cellprior=win._cp(r), **win._mcfg(s, r_type=r["machine_type"]))
        p1 = min(1 - 1e-4, max(1e-4, fp.pwin(win, r, s)))
        base[r["query_id"]] = [1 - p0] + [p0 * x for x in sz]; new[r["query_id"]] = [1 - p1] + [p1 * x for x in sz]
    print(f"validation predict (both): {time.time() - t0:.0f}s")
    b0 = points(pub, base, tune)[0]
    for name, p in (("FINAL (flat tau prior)", base), ("+ learned flip-time prior", new)):
        tp, tr = points(pub, p, tune); print(f"  {name:28s} tuning {tp:.3f} ({tp - b0:+.3f})  " + "  ".join(f"{c[:6]} {tr[c]:.4f}" for c in CONDITIONS))


if __name__ == "__main__":
    main()
