"""Competition-metric evaluation of FROZEN Method 5 baselines on the validation TUNING half (audit untouched).

  python -m method5_simulation.frozen_eval --data data/main

Trains B0 (type x signal prior) and B2 (boosting + last 8 outcomes) once on ALL train data (self-supervised late ticks +
labelled cutoff queries), predicts every validation query at its cutoff, and scores with the actual competition formula
(common.evaluate.points: squared error to the TRUE probabilities, per-condition ratios to the frozen reference).
S1 is scored from its pure-Python predictions (out/validation_s1.csv, written by bench.py) when present.
Validation targets are used only by the scorer; no model sees them. The audit half is never evaluated here.
"""
import argparse, csv, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from common.data import public_dir
from common.evaluate import folds, points, CONDITIONS
from method5_simulation import data as D, cv


def vectors(P):
    return [list(map(float, r)) for r in P]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, audit = folds(pub)
    M = D.load_machines(pub, "train"); X = D.examples(M, D.train_labels(pub)); V = D.load_machines(pub, "validation")
    Wq, Sq = zip(*[D.build(m["cats"], [m["cut"]], m["typ"], m["sig"]) for m in V])
    Xq = dict(W=np.concatenate(Wq), S=np.concatenate(Sq)); qids = [m["qid"] for m in V]
    key = lambda S: (S[:, :3].argmax(1) * 3 + (S[:, 3] + 1)).astype(int)
    prior = np.zeros((9, 5)); cnt = np.zeros(9)
    for k, y in zip(key(X["S"]), X["y"]):
        prior[k, y] += 1; cnt[k] += 1
    prior = (prior + 1) / (cnt[:, None] + 5)
    preds = {"B0 type x signal prior": prior[key(Xq["S"])]}
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=cv.SEED).fit(cv.tab(X, True), X["y"])
    preds["B2 boosting + last 8 outcomes"] = clf.predict_proba(cv.tab(Xq, True))
    os.makedirs("method5_simulation/out", exist_ok=True)
    for tag, key_ in (("b0", "B0 type x signal prior"), ("b2", "B2 boosting + last 8 outcomes")):   # saved for the diagnostics
        with open(f"method5_simulation/out/validation_{tag}.csv", "w") as f:
            f.write("query_id,p_0,p_1,p_2,p_5,p_20\n")
            for q_, row in zip(qids, preds[key_]):
                f.write(q_ + "," + ",".join(format(float(x), ".17g") for x in row) + "\n")
    for tag, label in (("s2", "S2 GRU observed-only (pure-Python)"), ("s1", "S1 GRU dense (pure-Python, ablation ref.)")):
        path = f"method5_simulation/out/validation_{tag}.csv"
        if os.path.exists(path):
            rows = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in csv.DictReader(open(path))}
            preds[label] = np.array([rows[q] for q in qids])
    print(f"{'candidate (frozen, trained on train only)':44s} {'tuning':>8s}  " + "  ".join(f"{c[:6]:>6s}" for c in CONDITIONS))
    for name, P in preds.items():
        pr = dict(zip(qids, vectors(P))); tp, tr = points(pub, pr, tune)
        print(f"{name:44s} {tp:8.3f}  " + "  ".join(f"{tr[c]:.4f}" for c in CONDITIONS))
    print(f"{'reference: Method 2 final (verified)':44s} {58.156:8.3f}  stable 0.0211  short 0.0617  drift 0.0257  shared 0.0242")
    print("organizer baseline = 30 points (all ratios 1.0). Audit half not evaluated (no candidate qualifies).")


if __name__ == "__main__":
    main()
