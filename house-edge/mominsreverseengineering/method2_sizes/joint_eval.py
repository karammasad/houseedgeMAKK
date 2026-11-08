"""Experiment 9 ablations on the validation TUNING half (no audit; no results.csv writes).

  python -m method2_sizes.joint_eval --data data/main

For the FINAL win gate (adi + size_split) and for adi's round-5 cell_signal prior (win-side alternative):
  one-way (= FINAL) | size->win feedback only | exact win->size only | full joint (both)
Reports tuning points, condition ratios, the WIN part of the loss (decompose.parts), how much the structure
posterior moves on short histories, and timing.
"""
import argparse, csv, math, os, time
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import folds, points, COLS, CONDITIONS
from method2_sizes.round2 import FINAL
from method2_sizes.model import Model
from method2_sizes.joint import joint_parts, assemble, lag_prior_rows
from method2_sizes.decompose import parts

ABL = [("one-way (current)", False, False), ("size->win feedback", True, False),
       ("exact win->size", False, True), ("full joint", True, True)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub); vq, vs = load_split(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    for gate_name, extra in (("FINAL gate (size_split)", {}), ("+ adi cell_signal (round 5)", {"cell_signal": True})):
        kw = dict(FINAL, win_kwargs=dict(FINAL["win_kwargs"], **extra))
        t0 = time.time(); m = Model(**kw).fit(pub); tfit = time.time() - t0
        rows = lag_prior_rows(m.win, m.sizes); preds = {n: {} for n, _, _ in ABL}
        shift = defaultdict(list); t0 = time.time()
        for r in vq:
            s = vs[r["machine_id"]]; P = joint_parts(m.win, m.sizes, rows, r, s)
            for n, fb, sa in ABL:
                p, post = assemble(m.sizes, rows, r, s, P, fb, sa); preds[n][r["query_id"]] = p
                if n == "full joint" and r["episode_id"] in tune:
                    shift[cond[r["query_id"]]].append(0.5 * sum(abs(x - y) for x, y in zip(post, P["post_s"])))
        tpred = time.time() - t0
        print(f"\n== {gate_name}: fit {tfit:.0f}s, all-ablation validation predict {tpred:.0f}s ==")
        print(f"  {'variant':22s} {'tuning':>7s} {'vs 1st':>7s}  " + "  ".join(f"{c[:6]:>6s}" for c in CONDITIONS) + "   WIN part (stable/short/drift/shared)")
        base = None
        for n, _, _ in ABL:
            tp, tr = points(pub, preds[n], tune); base = tp if base is None else base
            win = defaultdict(list)
            for k, p in preds[n].items():
                if qe := (k in cond):
                    pass
            for r in vq:
                if r["episode_id"] in tune:
                    w, *_ = parts(preds[n][r["query_id"]], T[r["query_id"]]); win[cond[r["query_id"]]].append(w)
            print(f"  {n:22s} {tp:7.3f} {tp - base:+7.3f}  " + "  ".join(f"{tr[c]:.4f}" for c in CONDITIONS) +
                  "   " + "/".join(f"{sum(win[c]) / len(win[c]):.4f}" for c in CONDITIONS))
        print("  structure-posterior shift from size feedback (total variation, tuning): " +
              "  ".join(f"{c[:6]} mean {sum(v) / len(v):.3f}, >0.1 on {sum(x > .1 for x in v)}/{len(v)}" for c, v in shift.items()))


if __name__ == "__main__":
    main()
