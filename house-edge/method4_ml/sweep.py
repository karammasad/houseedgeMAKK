"""Tuning-half sweep for Method 4 v2 (never touches the audit half). Owner: Person D.
One fit per (data, GBM) config; blend and size model are varied at predict time.
Usage (from house-edge/): python -m method4_ml.sweep --grid '{"K":[8,16],"K_short":[8,16]}'
"""
import argparse, itertools, json, time
from common.evaluate import folds, points
from method4_ml.model import Model

BLENDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SIZES = ("hmm", "npmle", "mix")


def run(pub, kw, tune):
    t0 = time.time(); m = Model(sizes="mix", **kw).fit(pub); fit_s = time.time() - t0
    rows = []; sm = dict(m.size_models)
    for sz in SIZES:
        m.size_models = {k: v for k, v in sm.items() if sz == "mix" or k == sz}
        for b in BLENDS:
            m.blend = b; pred = m.predict(pub, "validation"); tp, r = points(pub, pred, tune)
            rows.append((tp, sz, b, r))
    return rows, fit_s, m.n_train


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--grid", default="{}")
    a = ap.parse_args(); grid = json.loads(a.grid); tune, _ = folds(a.data)
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals)); rows, fs, n = run(a.data, kw, tune); rows.sort(key=lambda x: -x[0])
        tp, sz, b, r = rows[0]
        print(f"{json.dumps(kw)} n={n} fit {fs:.0f}s  best tuning {tp:.3f} (sizes={sz}, blend={b}) "
              + " ".join(f"{c[:5]} {v:.4f}" for c, v in r.items()), flush=True)
        for tp, sz, b, r in rows[1:4]:
            print(f"      {tp:.3f} sizes={sz} blend={b}")


if __name__ == "__main__":
    main()
