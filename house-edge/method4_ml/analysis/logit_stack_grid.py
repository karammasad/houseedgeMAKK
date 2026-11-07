"""l2 and blend-weight grid for the structured offset-logistic correction (selection on TUNING; audit shown as a check)."""
import numpy as np
exec(open("method4_ml/analysis/logit_stack_test.py").read().split("print(f\"{'model'")[0])
print("spec / l2 -> alone (tuning audit full)  |  blend weight on logistic: tuning / audit / full")
for name, spec in [("d", ["d"]), ("d+chg", ["d", "chg"]), ("d+short+chg+sig", ["d", "dshort", "chg", "sig"])]:
    Z = design(Xs, spec); Zq = design(Xq, spec)
    for l2 in (0.0, 0.1, 0.3, 1.0, 3.0):
        b = fit_offset_logistic(Z, ys, lg(Xs), l2 + 1e-9); pw = predict(Zq, b, lg(Xq)); a = score(pw)
        line = f"{name:16s} l2={l2:<4g} alone {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} |"
        for w in (0.3, 0.5, 0.7):
            e = score(w * pw + (1 - w) * tree_pw); line += f"  w={w}: {e[0]:.3f}/{e[1]:.3f}/{e[2]:.3f}"
        print(line, flush=True)
