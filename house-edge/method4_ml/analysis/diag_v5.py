"""Phase-1 diagnostics for v5 (analysis only: validation targets read here, never in fit/predict).
(a) oracle decomposition by condition; (b) residual learnability: episode-grouped 5-fold CV of a small GBM predicting
v5's errors from v5's own features (if CV R^2 ~ 0, the remaining error is not explained by what the model can see)."""
import csv, numpy as np
from collections import defaultdict
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
m = Model().fit(PUB); pred = m.predict(PUB, "validation"); tune, audit = folds(PUB)
def mix(pw_src, sz_src):
    return {k: [pw_src[k][0]] + [(1 - pw_src[k][0]) * x / max(1 - sz_src[k][0], 1e-12) for x in sz_src[k][1:]] for k in pred}
print("(a) oracle decomposition, FULL validation (ratios: stable short drift shared)")
for name, p in [("v5", pred), ("true P(win) + v5 sizes", mix(T, pred)), ("v5 P(win) + true sizes", mix(pred, T)), ("truth", T)]:
    tp, r = points(PUB, p, None); print(f"   {name:24s} {tp:.3f}  " + " ".join(f"{float(v):.4f}" for v in r.values()))
# (b) residual learnability on P(win) error
q, seq = load_split(PUB, "validation"); X, _, ids = m._features(PUB, "validation", queries=True); X = np.asarray(X, float)
pos = {k: i for i, k in enumerate(ids)}
y_w = np.array([(1 - pred[k][0]) - (1 - T[k][0]) for k in ids])           # P(win) error
y_s = np.array([pred[k][1] / max(1 - pred[k][0], 1e-9) - T[k][1] / max(1 - T[k][0], 1e-9) for k in ids])  # P(1x|win) error
eps = sorted({r["episode_id"] for r in q}); ep_of = {r["query_id"]: r["episode_id"] for r in q}
fold = {e: i % 5 for i, e in enumerate(eps)}
from sklearn.ensemble import HistGradientBoostingRegressor
for name, y in (("P(win) error", y_w), ("P(1x|win) error", y_s)):
    yhat = np.zeros_like(y)
    for f in range(5):
        tr = np.array([fold[ep_of[k]] != f for k in ids]); te = ~tr
        g = HistGradientBoostingRegressor(max_leaf_nodes=7, min_samples_leaf=30, learning_rate=.05, max_iter=200, random_state=0).fit(X[tr], y[tr])
        yhat[te] = g.predict(X[te])
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"(b) {name}: mean {y.mean():+.4f} (bias), sd {y.std():.4f}; episode-grouped 5-fold CV R^2 of residual model = {r2:+.3f}")
