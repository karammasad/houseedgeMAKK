"""Experiment 12: does exact marginalization over unobserved gate inputs add information on short logs?

  python -m method2_sizes.missfilter_eval PARAMS.json CACHE.json --data data/main

CACHE.json = cached FINAL predictions (cache_preds). Fits the filter's structure prior on TRAIN only. For every
validation short-history query: P_full (uses the sampled-tick outcomes via hidden-bit marginalization) vs P_ctrl
(the same filter with those ticks discarded, as the existing gates do). Reports (1) the standalone P(win) squared
error vs truth for FINAL / ctrl / full (analysis only), (2) the score when FINAL's P(win) on short logs is
corrected by the filter's own increment, P' = P_FINAL + (P_full - P_ctrl), tuning half only.
"""
import argparse, csv, json, os, pickle, sys, time
from common.data import load_split, public_dir
from common.evaluate import folds, points, COLS, CONDITIONS
from adi.model import is_short
from method2_sizes.missfilter import MissFilter


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("params"); ap.add_argument("cache"); ap.add_argument("--data", default="data/main")
    ap.add_argument("--G", type=int, default=12); a = ap.parse_args(); pub = public_dir(a.data); tune, _ = folds(pub)
    par = json.load(open(a.params)); C = json.load(open(a.cache))["final"]["validation"]
    tq, ts = load_split(pub, "train"); t0 = time.time(); mf = MissFilter(par["signal_priors"], G=a.G).fit(tq, ts)
    print(f"structure prior fit on train: {time.time() - t0:.0f}s; top hypotheses {[(round(p, 3), n) for p, n in mf.top(6)]}")
    vq, vs = load_split(pub, "validation")
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(os.path.join(pub, "validation", "targets.csv")))}
    t0 = time.time(); rows = []
    for r in vq:
        s = vs[r["machine_id"]]
        if is_short(s):
            rows.append((r, mf.pwin(r, s, True), mf.pwin(r, s, False)))
    print(f"filter on {len(rows)} validation short logs: {time.time() - t0:.0f}s")
    pf = lambda k: 1 - C[k][0] / sum(C[k])
    def mse(f, sel):
        v = [(f(r) - (1 - T[r['query_id']][0])) ** 2 for r, _, _ in rows if sel(r)]; return sum(v) / len(v)
    for name, sel in (("all short", lambda r: True), ("tuning half", lambda r: r["episode_id"] in tune)):
        print(f"\nP(win) squared error vs truth, {name} (n={sum(1 for r, _, _ in rows if sel(r))}):")
        print(f"  FINAL (adi gate)            {mse(lambda r: pf(r['query_id']), sel):.5f}")
        d = {r["query_id"]: (f, c) for r, f, c in rows}
        print(f"  filter, unknown ticks off   {mse(lambda r: d[r['query_id']][1], sel):.5f}")
        print(f"  filter, unknown ticks ON    {mse(lambda r: d[r['query_id']][0], sel):.5f}")
        print(f"  FINAL + (ON - off)          {mse(lambda r: min(.999, max(.001, pf(r['query_id']) + d[r['query_id']][0] - d[r['query_id']][1])), sel):.5f}")
    big = sorted(abs(f - c) for _, f, c in rows); print(f"\n|P_full - P_ctrl| quantiles: median {big[len(big) // 2]:.4f}, q90 {big[9 * len(big) // 10]:.4f}, max {big[-1]:.4f}")
    base = dict(C)
    for lam in (1.0,):
        new = dict(base)
        for r, f, c in rows:
            k = r["query_id"]; p = C[k]; pw = 1 - p[0] / sum(p); pw2 = min(.999, max(.001, pw + lam * (f - c)))
            new[k] = [1 - pw2] + [pw2 * x / (sum(p) - p[0]) for x in p[1:]]
        b = points(pub, base, tune); n = points(pub, new, tune)
        print(f"\nscore, tuning half: FINAL {b[0]:.3f} -> corrected {n[0]:.3f} ({n[0] - b[0]:+.3f});  short ratio {b[1]['short_history']:.4f} -> {n[1]['short_history']:.4f}")
        print("  conditions: " + "  ".join(f"{c[:6]} {b[1][c]:.4f}->{n[1][c]:.4f}" for c in CONDITIONS))


if __name__ == "__main__":
    main()
