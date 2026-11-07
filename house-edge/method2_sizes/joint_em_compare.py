"""Compare the current (best-lag) EB estimator with joint posterior-weighted EM for GateSizes.

  python -m method2_sizes.joint_em_compare --data data/main

Win gate (adi C5, as in FINAL) is fitted once and held fixed; only the size model's training procedure changes.
Population parameters are fitted on train; scores are on the validation TUNING half only (no audit access).
Does not write experiments/results.csv. Reports tuning points, the four tuning loss ratios, size-fit runtime,
learned tie weight / partition prior, and the predict-time posterior entropy over (size lag x shape).
"""
import argparse, csv, math, os, time
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import folds, points, CONDITIONS
from method2_sizes.round2 import FINAL
from method2_sizes.sizes import GateSizes
import importlib


def size_model(joint):
    return GateSizes(cell_prior="npmle", couple=True, couple_iters=20, tied=True, tie_weight="learn",
                     joint_em=joint)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub)
    tq, ts = load_split(pub, "train"); vq, vs = load_split(pub, "validation")
    with open(os.path.join(pub, "validation", "conditions.csv")) as f:
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}
    t0 = time.time()
    win = importlib.import_module(FINAL["win_model"] + ".model").Model(**FINAL["win_kwargs"]).fit(pub)
    wpred = win.predict(pub, "validation"); twin = time.time() - t0
    print(f"win gate (adi C5, fixed) fit+predict: {twin:.0f}s")
    rows = []
    for name, joint in (("best-lag EB (current default)", False), ("joint posterior-weighted EM", True)):
        t0 = time.time(); g = size_model(joint).fit(tq, ts); tfit = time.time() - t0
        pred, ent = {}, defaultdict(list)
        for r in vq:
            s = vs[r["machine_id"]]; p = wpred[r["query_id"]]; pw = 1 - p[0] / sum(p)
            pred[r["query_id"]] = [1 - pw] + [pw * x for x in g.predict(r, s)]
            if r["episode_id"] in tune:
                d = g.posterior_detail(r, s)
                h = -sum(x * math.log(x) for _, f, tt in d for x in (f, tt) if x > 0)
                ent[cond[r["query_id"]]].append(h)
        tp, tr = points(pub, pred, tune)
        rows.append((name, tp, tr, tfit, g, {c: sum(v) / len(v) for c, v in ent.items()}))
    print()
    print(f"{'estimator':32s} {'tuning':>7s}  " + "  ".join(f"{c[:6]:>6s}" for c in CONDITIONS) +
          "  size-fit   tie w   rho[big|1x|loss]      entropy(stable/short/drift/shared, nats)")
    for name, tp, tr, tfit, g, ent in rows:
        print(f"{name:32s} {tp:7.3f}  " + "  ".join(f"{tr[c]:.4f}" for c in CONDITIONS) +
              f"  {tfit:6.0f}s   {g.tie_weight:.3f}   {[round(x, 3) for x in g.part_prior]}   " +
              "/".join(f"{ent[c]:.3f}" for c in CONDITIONS))
    g = rows[1][4]
    if g.joint_trace:
        tr = g.joint_trace
        print("\njoint EM trace (train log-lik should rise monotonically):")
        for x in (tr[0], tr[1], tr[4], tr[9], tr[-1]):
            print(f"  iter {x['iter']:2d}  loglik {x['loglik']:.2f}  tie w {x['tie_weight']:.4f}  rho "
                  f"{[round(v, 3) for v in x['rho']]}  train posterior entropy {x['train_entropy']:.3f}")
        mono = all(b["loglik"] >= a["loglik"] - 1e-6 for a, b in zip(tr, tr[1:]))
        print("  monotone:", mono)
    print("\nlag prior g [none, 1..8]:")
    for name, *_, g, _ in [(r[0], r[4], r[4], r[5]) for r in rows]:
        print(f"  {name:32s} {[round(x, 3) for x in g.prior]}")


if __name__ == "__main__":
    main()
