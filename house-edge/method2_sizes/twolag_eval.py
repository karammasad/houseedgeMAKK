"""Experiment 8, Phase 3: compare size experts on the validation TUNING half (no audit).

  python -m method2_sizes.twolag_eval --data data/main

The win gate (adi C5, as in FINAL) is fitted once and held fixed; only the size model changes:
  A  current joint-EM GateSizes (final)          B  two-lag expert alone (pairs guided by method 1)
  C  Bayesian mixture A + two-lag (method 1)     D  mixture, two-lag restricted to method 1's top pairs
  E  mixture, two-lag pairs independent of method 1 (uniform over 28)
Reports tuning points, condition ratios, P(1x|win) error by condition and by observed win count, learned prior
weight v and mean posterior weight on the new expert, size-model fit+predict time, and peak memory.
Does not write experiments/results.csv.
"""
import argparse, csv, importlib, os, resource, time
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import folds, points, COLS, CONDITIONS
from method2_sizes.round2 import FINAL
from method2_sizes.sizes import GateSizes
from method2_sizes.twolag import TwoLagSizes


def base():
    return GateSizes(cell_prior="npmle", couple=True, couple_iters=20, tied=True, tie_weight="learn", joint_em=True)


VARIANTS = [("A  joint-EM GateSizes (final)", lambda: base()),
            ("B  two-lag alone (m1 pairs)", lambda: TwoLagSizes(base(), mode="alone", pairs="m1")),
            ("C  mixture, two-lag m1 pairs", lambda: TwoLagSizes(base(), mode="mix", pairs="m1")),
            ("D  mixture, m1 top pairs only", lambda: TwoLagSizes(base(), mode="mix", pairs="m1_only")),
            ("E  mixture, pairs indep. of m1", lambda: TwoLagSizes(base(), mode="mix", pairs="uniform"))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub)
    tq, ts = load_split(pub, "train"); vq, vs = load_split(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    t0 = time.time(); win = importlib.import_module(FINAL["win_model"] + ".model").Model(**FINAL["win_kwargs"]).fit(pub)
    wpred = win.predict(pub, "validation"); print(f"win gate fixed (adi C5): {time.time() - t0:.0f}s\n")
    res = []
    for name, make in VARIANTS:
        t0 = time.time(); m = make().fit(tq, ts); pred = {}; aerr = defaultdict(list); wb = defaultdict(list); wn = []
        for r in vq:
            s = vs[r["machine_id"]]; p = wpred[r["query_id"]]; pw = 1 - p[0] / sum(p); sz = m.predict(r, s)
            pred[r["query_id"]] = [1 - pw] + [pw * x for x in sz]
            if r["episode_id"] in tune:
                tw = 1 - T[r["query_id"]][0]; ta = T[r["query_id"]][1] / tw if tw > 1e-9 else sz[0]
                e = abs(sz[0] - ta); aerr[cond[r["query_id"]]].append(e)
                nw = sum(1 for v in s if v > 0); wb[0 if nw < 20 else 1 if nw < 40 else 2 if nw < 80 else 3].append(e)
                if isinstance(m, TwoLagSizes):
                    wn.append(m.weight_new(r, s))
        secs = time.time() - t0; tp, tr = points(pub, pred, tune)
        res.append((name, tp, tr, aerr, wb, getattr(m, "v", None), sum(wn) / len(wn) if wn else None, secs))
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)   # macOS reports bytes
    print(f"{'variant':32s} {'tuning':>7s} {'vs A':>7s}  " + "  ".join(f"{c[:6]:>6s}" for c in CONDITIONS) +
          "   v(prior)  post.mass  size fit+pred")
    for name, tp, tr, *_rest in res:
        v, pm, secs = _rest[2], _rest[3], _rest[4]
        print(f"{name:32s} {tp:7.3f} {tp - res[0][1]:+7.3f}  " + "  ".join(f"{tr[c]:.4f}" for c in CONDITIONS) +
              f"   {'-' if v is None else f'{v:.3f}':>8s}  {'-' if pm is None else f'{pm:.3f}':>9s}  {secs:6.0f}s")
    print("\nmean |P(1x|win) error| by condition (tuning):")
    for name, _, _, aerr, *_ in res:
        print(f"  {name:32s} " + "  ".join(f"{c[:6]} {sum(aerr[c]) / len(aerr[c]):.4f}" for c in CONDITIONS))
    print("\nmean |P(1x|win) error| by observed wins (<20 / 20-40 / 40-80 / 80+):")
    for name, _, _, _, wb, *_ in res:
        print(f"  {name:32s} " + "  ".join(f"{sum(wb[i]) / len(wb[i]):.4f} (n={len(wb[i])})" for i in range(4)))
    print(f"\npeak memory (whole process): {mem:.0f} MiB")


if __name__ == "__main__":
    main()
