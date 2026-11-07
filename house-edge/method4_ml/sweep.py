"""Tuning-half sweep for Method 4 v3 (never touches the audit half). Owner: Person D.
One fit per config (features and gate fit are cached); blend (and size model) vary at predict time.
Usage (from house-edge/): python -m method4_ml.sweep --grid '{"lr":[0.03,0.1],"max_leaf_nodes":[15,31]}'
Prints the best (blend, sizes) per config plus the full blend curve with --curve."""
import argparse, itertools, json, time
from common.evaluate import folds, points
from method4_ml.model import Model

BLENDS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def run(pub, kw, tune, sizes):
    t0 = time.time(); m = Model(**kw).fit(pub); fit_s = time.time() - t0; rows = []
    for b in BLENDS:
        m.blend = b; tp, r = points(pub, m.predict(pub, "validation"), tune); rows.append((tp, b, r))
    return rows, fit_s, m.n_train


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--grid", default="{}")
    ap.add_argument("--curve", action="store_true"); ap.add_argument("--sizes", default="gate")
    a = ap.parse_args(); grid = json.loads(a.grid); tune, _ = folds(a.data); keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals)); kw.setdefault("sizes", a.sizes)
        rows, fs, n = run(a.data, kw, tune, a.sizes); best = max(rows, key=lambda x: x[0])
        print(f"{json.dumps(kw)} n={n} fit {fs:.0f}s  best {best[0]:.3f} @blend={best[1]}  "
              + " ".join(f"{c[:5]} {v:.4f}" for c, v in best[2].items()), flush=True)
        if a.curve:
            print("   curve: " + "  ".join(f"{b}:{tp:.3f}" for tp, b, _ in rows), flush=True)


if __name__ == "__main__":
    main()
