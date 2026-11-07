"""Keep/revert check for one change (team rule): compare candidate B against baseline A on validation.

  python -m method2_sizes.keepcheck --data data/main --a '{"method":"method2_sizes","kwargs":{...}}' \
                                    --b '{"method":"method2_sizes","kwargs":{...}}'

Reports tuning / audit / full points, the four full-validation condition ratios, runtime, and a paired
per-episode bootstrap (resampling episodes within each condition, both models on the same draw) of the
improvement in R. KEEP needs: tuning and audit both up, bootstrap P(gain > 0) >= 0.95, >= 3 of 4 conditions
better and none worse by more than 5% relative, runtime < 300 s. Diagnosis tool: reads validation targets
only here, never inside fit/predict. Bootstrap seed fixed.
"""
import argparse, csv, importlib, json, os, random, time
from collections import defaultdict
from common.data import public_dir
from common.evaluate import folds, points, COLS, CONDITIONS


def run(spec, pub):
    t0 = time.time()
    m = importlib.import_module(spec["method"] + ".model").Model(**spec.get("kwargs", {})).fit(pub)
    pred = m.predict(pub, "validation")
    return pred, time.time() - t0


def episode_errors(pub, pred):
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    ref = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("reference.csv")}
    e = defaultdict(lambda: defaultdict(list)); b = defaultdict(lambda: defaultdict(list))
    for t in rd("targets.csv"):
        k = t["query_id"]; tr = [float(t[c]) for c in COLS]; p = pred[k]; s = sum(p)
        e[cond[k]][t["episode_id"]].append(sum((x / s - y) ** 2 for x, y in zip(p, tr)))
        b[cond[k]][t["episode_id"]].append(sum((x - y) ** 2 for x, y in zip(ref[k], tr)))
    E = {c: {ep: sum(v) / len(v) for ep, v in d.items()} for c, d in e.items()}
    B = {c: {ep: sum(v) / len(v) for ep, v in d.items()} for c, d in b.items()}
    return E, B


def R_of(E, B, draw):
    return sum(sum(E[c][ep] for ep in draw[c]) / max(1e-9, sum(B[c][ep] for ep in draw[c])) for c in CONDITIONS) / 4


def compare(pub, pa, pb, ta, tb, reps=2000):
    tune, audit = folds(pub)
    out = {}
    for name, pred, secs in (("A", pa, ta), ("B", pb, tb)):
        tp = points(pub, pred, tune)[0]; ap = points(pub, pred, audit)[0]; fp, fr = points(pub, pred)
        out[name] = dict(tuning=tp, audit=ap, full=fp, ratios=fr, secs=secs)
    Ea, B = episode_errors(pub, pa); Eb, _ = episode_errors(pub, pb)
    rng = random.Random(12345); wins = 0; gains = []
    eps = {c: sorted(B[c]) for c in CONDITIONS}
    for _ in range(reps):
        draw = {c: [rng.choice(eps[c]) for _ in eps[c]] for c in CONDITIONS}
        g = R_of(Ea, B, draw) - R_of(Eb, B, draw); gains.append(g); wins += g > 0
    pgain = wins / reps
    ra, rb = out["A"]["ratios"], out["B"]["ratios"]
    better = sum(rb[c] < ra[c] for c in CONDITIONS); worse = any(rb[c] > ra[c] * 1.05 for c in CONDITIONS)
    keep = (out["B"]["tuning"] > out["A"]["tuning"] and out["B"]["audit"] > out["A"]["audit"] and pgain >= .95
            and better >= 3 and not worse and out["B"]["secs"] < 300)
    return out, pgain, better, worse, keep


def report(out, pgain, better, worse, keep):
    for n in ("A", "B"):
        o = out[n]
        print(f"  {n}: tuning {o['tuning']:.3f}  audit {o['audit']:.3f}  full {o['full']:.3f}  " +
              "  ".join(f"{c[:6]} {o['ratios'][c]:.4f}" for c in CONDITIONS) + f"  ({o['secs']:.0f}s)")
    print(f"  bootstrap P(B better) = {pgain:.3f};  conditions improved {better}/4;  any >5% worse: {worse}")
    print("  VERDICT:", "KEEP" if keep else "REVERT")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main")
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    a = ap.parse_args(); pub = public_dir(a.data)
    pa, ta = run(json.loads(a.a), pub); pb, tb = run(json.loads(a.b), pub)
    report(*compare(pub, pa, pb, ta, tb))


if __name__ == "__main__":
    main()
