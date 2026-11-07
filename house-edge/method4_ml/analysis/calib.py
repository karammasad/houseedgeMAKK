"""Calibration of the gate's P(win) by condition: truth ~ pred bins, slope of truth on pred (1 = calibrated, <1 = too extreme)."""
import csv
from collections import defaultdict
import numpy as np
from common.data import load_split
from method4_ml.model import Model
PUB = "data/main"
T = {r["query_id"]: 1 - float(r["p_0"]) for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/validation/conditions.csv"))}
m = Model(blend=0).fit(PUB); q, seq = load_split(PUB, "validation"); pw = m.gm.win_probs(q, seq)
for c in ("stable", "short_history", "drift", "shared_change"):
    for typ in ("steady", "balanced", "jackpot"):
        ks = [r["query_id"] for r in q if C[r["query_id"]] == c and r["machine_type"] == typ]
        p = np.array([pw[k] for k in ks]); t = np.array([T[k] for k in ks])
        slope = np.cov(p, t)[0, 1] / np.var(p, ddof=1)
        print(f"{c:13s} {typ:8s} n={len(ks):3d} mse {np.mean((p - t) ** 2):.4f}  slope {slope:.2f}  bias {np.mean(p - t):+.3f}")
