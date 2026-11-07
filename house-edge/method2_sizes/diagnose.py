"""Where does Method 2 lose points? Error decomposition on the validation TUNING half (diagnosis only;
reads targets here, never inside fit/predict).

  python -m method2_sizes.diagnose --data data/main --kwargs '{"sizes":"lag","lag_strength":100,"npmle_half_life":64}'

Splits each prediction into win probability pw = 1 - p0 and sizes s = p[1:] / pw, then swaps in the truth
for one part at a time to see how many points each part is costing.
"""
import argparse, csv, json, os
from collections import defaultdict
from common.data import public_dir, load_split
from common.evaluate import folds, points, COLS
from method2_sizes.model import Model


def split(p):
    pw = 1 - p[0]
    return pw, ([x / pw for x in p[1:]] if pw > 1e-12 else [0.25] * 4)


def join(pw, s):
    return [1 - pw] + [pw * x for x in s]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--kwargs", default="{}")
    a = ap.parse_args(); pub = public_dir(a.data)
    pred = Model(**json.loads(a.kwargs)).fit(pub).predict(pub, "validation")
    with open(os.path.join(pub, "validation", "targets.csv")) as f:
        truth = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(f)}
    tune, _ = folds(pub)
    variants = {
        "model": pred,
        "true sizes (model win prob)": {k: join(split(p)[0], split(truth[k])[1]) for k, p in pred.items()},
        "true win prob (model sizes)": {k: join(split(truth[k])[0], split(p)[1]) for k, p in pred.items()},
        "both true": truth,
    }
    print("== Points on tuning half when one component is replaced by the truth ==")
    for name, v in variants.items():
        pt, r = points(pub, v, tune)
        print(f"  {name:30s} {pt:7.3f}   " + "  ".join(f"{c[:6]} {x:.3f}" for c, x in r.items()))

    q, seq = load_split(pub, "validation"); qm = {r["query_id"]: r for r in q}
    with open(os.path.join(pub, "validation", "conditions.csv")) as f:
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}
    print("\n== Size error (weighted by true pw^2, i.e. its share of squared error) by type x condition ==")
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for k, p in pred.items():
        if qm[k]["episode_id"] not in tune:
            continue
        pw_t, s_t = split(truth[k]); pw_m, s_m = split(p)
        e_size = pw_t ** 2 * sum((x - y) ** 2 for x, y in zip(s_m, s_t))
        e_all = sum((x - y) ** 2 for x, y in zip(p, truth[k]))
        key = (qm[k]["machine_type"], cond[k]); agg[key][0] += e_size; agg[key][1] += e_all; agg[key][2] += 1
    for key in sorted(agg):
        e, t, n = agg[key]; print(f"  {key[0]:9s} {key[1]:14s} n={n:3d}  size err/query {e / n:.4f}  total err/query {t / n:.4f}")

    print("\n== Do true sizes depend on the true win probability? (corr of true P(1x|win) with true pw, per type) ==")
    by = defaultdict(list)
    for k in pred:
        if qm[k]["episode_id"] in tune:
            pw_t, s_t = split(truth[k]); by[qm[k]["machine_type"]].append((pw_t, s_t, split(pred[k])[1]))
    for t, rows in sorted(by.items()):
        def corr(x, y):
            mx, my = sum(x) / len(x), sum(y) / len(y)
            sx = sum((a - mx) ** 2 for a in x) ** .5; sy = sum((b - my) ** 2 for b in y) ** .5
            return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy + 1e-12)
        pw = [r[0] for r in rows]
        line = [f"P({v}x|win) corr w/ pw {corr(pw, [r[1][i] for r in rows]):+.2f}" for i, v in enumerate((1, 2, 5, 20))]
        fit = [f"{v}x: R2-ish corr(model,true) {corr([r[2][i] for r in rows], [r[1][i] for r in rows]):+.2f}" for i, v in enumerate((1, 2, 5, 20))]
        print(f"  {t}: " + ", ".join(line)); print("      " + ", ".join(fit))


if __name__ == "__main__":
    main()
