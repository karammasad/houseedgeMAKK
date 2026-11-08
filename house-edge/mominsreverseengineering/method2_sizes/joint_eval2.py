"""Experiment 9b: two predeclared feedback variants (tuning half only). Win gate = FINAL + adi cell_signal.
  python -m method2_sizes.joint_eval2 --data data/main
"""
import argparse, time
from common.data import load_split, public_dir
from common.evaluate import folds, points, CONDITIONS
from method2_sizes.round2 import FINAL
from method2_sizes.model import Model
from method2_sizes.joint import joint_parts, assemble, lag_prior_rows
from adi.model import is_short


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub); vq, vs = load_split(pub, "validation")
    kw = dict(FINAL, win_kwargs=dict(FINAL["win_kwargs"], cell_signal=True))
    m = Model(**kw).fit(pub); rows = lag_prior_rows(m.win, m.sizes)
    V = {"one-way (FINAL + cell_signal)": {}, "feedback, all logs, beta 1": {}, "feedback, short logs only": {},
         "feedback, all logs, beta 0.5": {}}
    for r in vq:
        s = vs[r["machine_id"]]; P = joint_parts(m.win, m.sizes, rows, r, s); k = r["query_id"]
        V["one-way (FINAL + cell_signal)"][k] = assemble(m.sizes, rows, r, s, P, False, False)[0]
        V["feedback, all logs, beta 1"][k] = assemble(m.sizes, rows, r, s, P, True, False)[0]
        V["feedback, short logs only"][k] = assemble(m.sizes, rows, r, s, P, is_short(s), False)[0]
        V["feedback, all logs, beta 0.5"][k] = assemble(m.sizes, rows, r, s, P, True, False, .5)[0]
    base = None
    for n, p in V.items():
        tp, tr = points(pub, p, tune); base = tp if base is None else base
        print(f"  {n:32s} tuning {tp:.3f} ({tp - base:+.3f})  " + "  ".join(f"{c[:6]} {tr[c]:.4f}" for c in CONDITIONS))


if __name__ == "__main__":
    main()
