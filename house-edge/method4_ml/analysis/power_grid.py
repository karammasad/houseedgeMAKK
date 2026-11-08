"""Grid blend / correction choices on the power harness: select on validation-history ticks, confirm on test-history ticks."""
import numpy as np, itertools
from method4_ml.model import Model
from method4_ml.logit_stack import design, fit_offset_logistic, predict as lrp
from method4_ml.analysis.power_harness import load_held, PUB
m = Model(splits="train").fit(PUB)
held = load_held(m); X = np.vstack([h[2] for h in held]); y = np.concatenate([h[3] for h in held])
tag = np.concatenate([[h[0]] * len(h[2]) for h in held]); short = X[:, 23] <= 64
Xt, yt = [], []
for bag in range(m.bags):
    m.tick_seed = bag; a, b, _ = m._features(PUB, "train"); Xt.append(np.asarray(a, float)); yt += [int(v > 0) for v in b]
Xt = np.vstack(Xt); yt = np.array(yt)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
tree = np.mean([c.predict_proba(X)[:, 1] for c in m.clfs], axis=0); g = X[:, 0]
def br(p, msk=None):
    e = (np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2
    return 1e3 * e.mean() if msk is None else 1e3 * e[msk].mean()
V, T = tag == "validation", tag == "test"
print("gate %.3f  tree(raw) %.3f" % (br(g), br(tree)))
res = []
for spec in (["d", "chg"], ["d"], ["d", "chg", "sig"], ["d", "chg", "short"], ["d", "dshort", "chg", "sig", "curv", "ep", "short"]):
    for l2 in (1.0, 30.0):
        beta = fit_offset_logistic(design(Xt, spec), yt, lg(Xt[:, 0]), l2); lr = lrp(design(X, spec), beta, lg(X[:, 0]))
        for bl in (0.4, 0.7, 1.0):
            mix = bl * tree + (1 - bl) * g
            for w in (0.0, 0.25, 0.5):
                p = w * lr + (1 - w) * mix; res.append((br(p, V), br(p, T), br(p), "+".join(spec), l2, bl, w))
res.sort()
print("top 12 by VALIDATION-history Brier (x1e3) | test | all | spec l2 tree_blend lr_weight")
for r in res[:12]: print("%.3f %.3f %.3f  %s l2=%g blend=%.1f w=%.2f" % r)
print("v7 setting:", [("%.3f %.3f %.3f" % r[:3]) for r in res if r[3] == "d+chg" and r[4] == 1.0 and r[5] == 0.4 and r[6] == 0.5])
print("v6 setting:", [("%.3f %.3f %.3f" % r[:3]) for r in res if r[3] == "d+chg" and r[4] == 1.0 and r[5] == 0.4 and r[6] == 0.0])
