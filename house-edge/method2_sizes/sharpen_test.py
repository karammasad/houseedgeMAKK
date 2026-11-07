"""Is v5 / FINAL under-confident on P(win)? One-parameter logit sharpening test.

  python -m method2_sizes.sharpen_test CACHE.json --data data/main

CACHE.json holds predictions from cache_preds (v5 and FINAL; validation + test).
p' = sigmoid(k * logit(p)); the sizes split is untouched, only P(win) moves. k is fitted on the validation TUNING
half only (grid 0.8..1.6), then scored once on the AUDIT half as the predeclared confirmation. k = 1 is identity.
Diagnostic of calibration, not a final model: a k fitted on validation truth must be learnable from train to ship.
"""
import argparse, json, math, sys
from common.data import public_dir
from common.evaluate import folds, points, CONDITIONS


def sharpen(pred, k):
    out = {}
    for q, p in pred.items():
        s = sum(p); p0 = p[0] / s; pw = min(1 - 1e-6, max(1e-6, 1 - p0))
        pw2 = 1 / (1 + math.exp(-k * math.log(pw / (1 - pw))))
        out[q] = [1 - pw2] + [pw2 * x / (s - p[0]) for x in p[1:]]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--data", default="data/main")
    a = ap.parse_args(); pub = public_dir(a.data); tune, audit = folds(pub)
    C = json.load(open(a.cache)); ks = [0.8 + .1 * i for i in range(9)]
    for name in ("v5", "final"):
        pred = C[name]["validation"]
        print(f"\n== {name}: tuning-half sweep of k (logit sharpening of P(win)) ==")
        res = {}
        for k in ks:
            tp, tr = points(pub, sharpen(pred, k), tune); res[k] = tp
            print(f"  k={k:.1f}  tuning {tp:.3f}  " + "  ".join(f"{c[:6]} {tr[c]:.4f}" for c in CONDITIONS))
        kb = max(res, key=res.get)
        ta, _ = points(pub, pred, audit); tk, _ = points(pub, sharpen(pred, kb), audit)
        fa, _ = points(pub, pred); fk, _ = points(pub, sharpen(pred, kb))
        print(f"  best k on tuning = {kb:.1f}: tuning {res[kb]:.3f} ({res[kb] - res[1.0]:+.3f})")
        print(f"  AUDIT (confirmation, k fixed from tuning): {ta:.3f} -> {tk:.3f} ({tk - ta:+.3f});  full {fa:.3f} -> {fk:.3f} ({fk - fa:+.3f})")


if __name__ == "__main__":
    main()
