"""Size model vs the machine's empirical 1x share among wins: binned mean pred vs mean truth (validation, diagnosis)."""
import csv, numpy as np
from collections import defaultdict
from common.data import load_split
from method4_ml.model import Model
PUB = "data/main"
COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
T = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
Cn = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/validation/conditions.csv"))}
m = Model().fit(PUB); pred = m.predict(PUB, "validation"); q, seq = load_split(PUB, "validation")
rows = []
for r in q:
    s = seq[r["machine_id"]]; k = r["query_id"]; w = [x for x in s if x > 0]
    e1 = sum(x == 1 for x in w) / max(len(w), 1)
    p1 = pred[k][1] / (1 - pred[k][0]); t1 = T[k][1] / (1 - T[k][0])
    rows.append((r["machine_type"], Cn[k], len(w), e1, p1, t1))
for typ in ("steady", "balanced", "jackpot"):
    R = [x for x in rows if x[0] == typ]
    e = np.array([x[3] for x in R]); p = np.array([x[4] for x in R]); t = np.array([x[5] for x in R])
    print(f"{typ}: corr(err, emp share) {np.corrcoef(p - t, e)[0, 1]:+.2f}; slope truth~pred {np.cov(p, t)[0, 1] / np.var(p, ddof=1):.2f}; mse {np.mean((p - t) ** 2):.4f}")
    for lo, hi in ((0, .3), (.3, .45), (.45, .6), (.6, .75), (.75, 1.01)):
        sel = (e >= lo) & (e < hi)
        if sel.sum() >= 8:
            print(f"    emp 1x share [{lo:.2f},{hi:.2f}) n={sel.sum():3d}  mean pred {p[sel].mean():.3f}  mean truth {t[sel].mean():.3f}  emp {e[sel].mean():.3f}")
