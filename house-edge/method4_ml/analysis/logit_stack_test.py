"""Structured offset-logistic corrections vs the tree stacker and their ensemble (tuning/audit/full; sizes fixed)."""
import numpy as np
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
from method4_ml.logit_stack import design, fit_offset_logistic, predict, COL, check_layout
PUB = "data/main"; tune, audit = folds(PUB)
m = Model().fit(PUB); base = m.predict(PUB, "validation")
Xq, _, ids = m._features(PUB, "validation", queries=True); Xq = np.asarray(Xq, float)
szkey = next(iter(m._sz)); SZ = m._sz[szkey]
Xs, ys = [], []
for bag in range(m.bags):
    m.tick_seed = bag
    for s in ("train", "validation", "test"):
        a, b, _ = m._features(PUB, s); Xs += a; ys += [int(v > 0) for v in b]
Xs = np.asarray(Xs, float); ys = np.array(ys); check_layout(Xs); check_layout(Xq)
tree_pw = np.array([1 - base[k][0] for k in ids])
def score(pw):
    out = {k: [1 - p] + [p * x for x in SZ[k]] for k, p in zip(ids, np.clip(pw, 1e-6, 1 - 1e-6))}
    return points(PUB, out, tune)[0], points(PUB, out, audit)[0], points(PUB, out, None)[0]
lg = lambda X: np.log(np.clip(X[:, 0], 1e-4, 1 - 1e-4) / (1 - np.clip(X[:, 0], 1e-4, 1 - 1e-4)))
print(f"{'model':38s} tuning  audit   full   (n_train {len(ys)})")
t = score(tree_pw); print(f"{'v6 tree stacker (bag 5)':38s} {t[0]:.3f} {t[1]:.3f} {t[2]:.3f}")
g = score(Xq[:, 0]); print(f"{'gate only':38s} {g[0]:.3f} {g[1]:.3f} {g[2]:.3f}")
res = {}
for name, spec in [("adi 6-term (d)", ["d"]), ("d + short", ["d", "dshort"]), ("d + chg", ["d", "chg"]), ("d + curv", ["d", "curv"]),
                   ("d + signal", ["d", "sig"]), ("d + episode", ["d", "ep"]), ("d + short + chg + sig", ["d", "dshort", "chg", "sig"]),
                   ("all terms", ["d", "dshort", "chg", "curv", "sig", "ep", "short"])]:
    for l2 in (1.0, 30.0):
        Z = design(Xs, spec); b = fit_offset_logistic(Z, ys, lg(Xs), l2)
        pw = predict(design(Xq, spec), b, lg(Xq)); res[(name, l2)] = pw; s_ = score(pw)
        e = score(0.5 * pw + 0.5 * tree_pw)
        print(f"{name + f' (l2={l2:g})':38s} {s_[0]:.3f} {s_[1]:.3f} {s_[2]:.3f}   | 50/50 with trees: {e[0]:.3f} {e[1]:.3f} {e[2]:.3f}", flush=True)
