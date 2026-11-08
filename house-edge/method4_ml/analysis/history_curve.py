"""Ceiling study (2026-11-08): how v8's error depends on how much history it sees (learning curve over history length).

Long validation logs (stable / drift / shared_change) are truncated to their last N ticks (earlier ticks set missing); the true
probabilities at the cutoff are unchanged (they belong to the simulator, not to what we observe), so error(N) is a clean
learning curve. Short logs are left as they are. The fitted v8 model (fit on data/main, cached) predicts each masked copy.
Fit per condition: err(N) = a + b / N on the long-log points -> b/N_full = estimation error that more history would remove,
a = what more history of the SAME kind would not remove (model-based extrapolation, not a theoretical ceiling).
Diagnosis only: reads validation targets outside fit/predict.
"""
import csv, os, shutil
from collections import defaultdict
import numpy as np
from common.data import load_split
from common.evaluate import points, _rows, COLS, CONDITIONS
from method4_ml.model import Model

PUB = "data/main"; NS = (40, 60, 90, 130, None)
OUT = os.path.join(os.path.dirname(__file__), "..", "cache", "history_curve")


def masked_pub(N):
    d = os.path.join(OUT, f"N{N}"); v = os.path.join(d, "validation")
    if os.path.exists(os.path.join(v, "history.csv")):
        return d
    os.makedirs(v, exist_ok=True)
    for f in ("queries.csv", "targets.csv", "conditions.csv", "reference.csv"):
        shutil.copy(os.path.join(PUB, "validation", f), v)
    cond = {r["query_id"]: r["condition"] for r in _rows(os.path.join(PUB, "validation", "conditions.csv"))}
    q, _ = load_split(PUB, "validation"); keep_from = {}
    for r in q:
        keep_from[r["machine_id"]] = r["cutoff"] - N if cond[r["query_id"]] != "short_history" else 0
    with open(os.path.join(PUB, "validation", "history.csv"), newline="") as f, open(os.path.join(v, "history.csv"), "w", newline="") as g:
        rd = csv.DictReader(f); w = csv.DictWriter(g, rd.fieldnames); w.writeheader()
        for row in rd:
            if int(row["tick"]) >= keep_from[row["machine_id"]]:
                w.writerow(row)
    return d


def cond_err(pub, pred):
    T = _rows(os.path.join(pub, "validation", "targets.csv")); C = {r["query_id"]: r["condition"] for r in _rows(os.path.join(pub, "validation", "conditions.csv"))}
    e = defaultdict(lambda: defaultdict(list))
    for t in T:
        e[C[t["query_id"]]][t["episode_id"]].append(sum((x - float(t[k])) ** 2 for x, k in zip(pred[t["query_id"]], COLS)))
    return {c: np.mean([np.mean(v) for v in e[c].values()]) for c in CONDITIONS}


def main():
    m = Model().fit(PUB); res = {}
    q, seq = load_split(PUB, "validation"); full_len = np.mean([r["cutoff"] for r in q])
    for N in NS:
        pub = PUB if N is None else masked_pub(N); pred = m.predict(pub, "validation")
        res[N] = (points(pub, pred)[0], cond_err(pub, pred))
        print(f"N={str(N or 'full'):>4s}: points {res[N][0]:.3f}  " + "  ".join(f"{c} {v:.5f}" for c, v in res[N][1].items()), flush=True)
    print("\nfit err(N) = a + b/N per long condition (N_full = mean cutoff %.0f):" % full_len)
    for c in ("stable", "drift", "shared_change"):
        x = np.array([1 / (N or full_len) for N in NS]); y = np.array([res[N][1][c] for N in NS])
        b, a = np.polyfit(x, y, 1); est = b / full_len; cur = res[None][1][c]
        print(f"  {c:14s} current {cur:.5f} = a {a:.5f} + estimation b/N {est:.5f} ({est / cur:.0%} of current); "
              f"doubling history would remove ~{(b / full_len - b / (2 * full_len)) / cur:.0%}")


if __name__ == "__main__":
    main()
