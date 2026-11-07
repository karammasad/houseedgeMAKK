"""Product-of-marginals bias. pred p_i = pw * s_i (separate posterior means). If pw and s_i depend on the same latent size state,
the posterior mean of the PRODUCT is E[pw]E[s_i] + Cov(pw, s_i). Diagnostic (validation truth): regress the class-i residual
(true - pred) on u_i = (pw - type mean pw)(s_i - type mean s_i), per type. Slope ~ +1 with R^2 > 0 => a covariance term is missing."""
import csv, numpy as np
from collections import defaultdict
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"; tune, audit = folds(PUB)
COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
T = {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
m = Model().fit(PUB); pred = m.predict(PUB, "validation")
from common.data import load_split
q, _ = load_split(PUB, "validation"); ty = {r["query_id"]: r["machine_type"] for r in q}
P = {k: np.array(v) for k, v in pred.items()}
pw = {k: 1 - P[k][0] for k in P}; s = {k: P[k][1:] / max(pw[k], 1e-9) for k in P}
mpw = {t: np.mean([pw[k] for k in P if ty[k] == t]) for t in ("steady", "balanced", "jackpot")}
ms = {t: np.mean([s[k] for k in P if ty[k] == t], axis=0) for t in mpw}
print("class  type      slope(resid~u)   R^2    corr")
for i, name in enumerate(("1x", "2x", "5x", "20x")):
    for t in mpw:
        ks = [k for k in P if ty[k] == t]
        u = np.array([(pw[k] - mpw[t]) * (s[k][i] - ms[t][i]) for k in ks]); r = np.array([T[k][i + 1] - P[k][i + 1] for k in ks])
        sl = np.cov(u, r)[0, 1] / np.var(u, ddof=1); c = np.corrcoef(u, r)[0, 1]
        print(f"{name:5s}  {t:9s} {sl:+8.2f}      {c * c:5.3f}  {c:+.2f}")

# optimistic upper bound: apply the in-sample fitted per-(class, type) correction and renormalize
new = {k: P[k].copy() for k in P}
for i in range(4):
    for t in mpw:
        ks = [k for k in P if ty[k] == t]
        u = np.array([(pw[k] - mpw[t]) * (s[k][i] - ms[t][i]) for k in ks]); r = np.array([T[k][i + 1] - P[k][i + 1] for k in ks])
        sl = np.cov(u, r)[0, 1] / np.var(u, ddof=1)
        for k, uu in zip(ks, u): new[k][i + 1] = max(new[k][i + 1] + sl * uu, 1e-6)
for k in new: new[k] = list(new[k] / new[k].sum())
print(f"\nin-sample upper bound of the covariance correction: v7 full {points(PUB, pred, None)[0]:.3f} -> {points(PUB, new, None)[0]:.3f}")
