"""Does RAW history contain information v5 does not use? Episode-grouped 5-fold CV of a GBM predicting v5's errors from raw
history (payouts at lags 1..60, counts, neighbours' last 20 payouts) plus v5's own outputs. Repeated over 5 fold seeds.
R^2 clearly > 0 => a raw-sequence learner could add signal; ~0 or < 0 => that class of models is unlikely to help."""
import csv, random, numpy as np
from common.data import load_split
from method4_ml.model import Model
from sklearn.ensemble import HistGradientBoostingRegressor
PUB = "data/main"
COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
T = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
m = Model().fit(PUB); pred = m.predict(PUB, "validation"); q, seq = load_split(PUB, "validation")
CAT = {-1: -1, 0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
eps = {}
for r in q: eps.setdefault(r["episode_id"], []).append(r)
X, yw, ys, ep, ids = [], [], [], [], []
for r in q:
    s = seq[r["machine_id"]]; cut = r["cutoff"]; k = r["query_id"]
    f = [CAT[s[cut - l]] if cut - l >= 0 else -1 for l in range(1, 61)]
    f += [sum(1 for x in s if x == v) for v in (0, 1, 2, 5, 20)] + [sum(1 for x in s if x >= 0)]
    for o in eps[r["episode_id"]]:
        if o is r: continue
        so = seq[o["machine_id"]]; f += [sum(1 for x in so[-20:] if x > 0), sum(1 for x in so[-20:] if x == 1)]
    f += [("steady", "balanced", "jackpot").index(r["machine_type"]), r["signal"]] + pred[k]
    X.append(f); ids.append(k); ep.append(r["episode_id"])
    yw.append((1 - pred[k][0]) - (1 - T[k][0]))
    ys.append(pred[k][1] / max(1 - pred[k][0], 1e-9) - T[k][1] / max(1 - T[k][0], 1e-9))
X = np.array(X, float); E = sorted(set(ep))
for name, y in (("P(win) error", np.array(yw)), ("P(1x|win) error", np.array(ys))):
    r2s = []
    for seed in range(5):
        rng = random.Random(seed); order = E[:]; rng.shuffle(order); fold = {e: i % 5 for i, e in enumerate(order)}
        f = np.array([fold[e] for e in ep]); yhat = np.zeros_like(y)
        for k in range(5):
            g = HistGradientBoostingRegressor(max_leaf_nodes=7, min_samples_leaf=30, learning_rate=.05, max_iter=150, random_state=0)
            g.fit(X[f != k], y[f != k]); yhat[f == k] = g.predict(X[f == k])
        r2s.append(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    print(f"{name}: CV R^2 over 5 fold seeds = {np.mean(r2s):+.3f} (min {min(r2s):+.3f}, max {max(r2s):+.3f})")

# which raw-feature group carries the size signal? drop-group CV R^2 (5 fold seeds)
groups = {"lags 1-8": range(0, 8), "lags 9-20": range(8, 20), "lags 21-60": range(20, 60), "counts": range(60, 66),
          "neighbours": range(66, 80), "type/signal": range(80, 82), "v5 outputs": range(82, 87)}
y = np.array(ys)
def cvr2(cols):
    r2s = []
    for seed in range(5):
        rng = random.Random(seed); order = E[:]; rng.shuffle(order); fold = {e: i % 5 for i, e in enumerate(order)}
        f = np.array([fold[e] for e in ep]); yhat = np.zeros_like(y)
        for k in range(5):
            g = HistGradientBoostingRegressor(max_leaf_nodes=7, min_samples_leaf=30, learning_rate=.05, max_iter=150, random_state=0)
            g.fit(X[f != k][:, cols], y[f != k]); yhat[f == k] = g.predict(X[f == k][:, cols])
        r2s.append(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.mean(r2s)
allc = list(range(X.shape[1])); print("size residual, all features:", round(cvr2(allc), 3))
for g, idx in groups.items():
    print(f"   without {g:12s}: {cvr2([c for c in allc if c not in set(idx)]):+.3f}   only {g:12s}: {cvr2(list(idx)):+.3f}")
