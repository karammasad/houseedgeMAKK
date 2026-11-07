"""Fair order ablation S3b (run AFTER the main CV folds finish).

  python -m method5_simulation.cv_ablate --fold k      -> results/cv_ablate_fold{k}.npz  (same held-out ordering as cv.py)

S3  (cv.py) permutes whole window columns, i.e. the category AND its observation flag move together. It therefore
    removes (a) the temporal order of outcomes AND (b) WHERE the unobserved ticks sit (the gap structure); it keeps only
    the multiset of (category, observed) pairs. That conflates order with gaps.
S3b (here) permutes only the CATEGORIES among the observed positions of each window; the observation mask is held
    fixed. It preserves exactly: the mask and gap pattern, the number of observed ticks, the category counts and the
    position of every unobserved tick. It removes ONLY the order in which the observed outcomes occurred.
For long logs (no unobserved ticks in the window) S3 and S3b coincide; they differ only on short logs.
"""
import argparse, os, time
import numpy as np
import torch
from common.data import public_dir
from method5_simulation import data as D, cv


def shuffle_obs(W, seed):
    rng = np.random.default_rng(seed); W2 = W.copy()
    for i in range(len(W)):
        o = np.nonzero(W[i] >= 0)[0]
        if len(o) > 1:
            W2[i, o] = W[i, o][rng.permutation(len(o))]
    return W2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--data", default="data/main")
    a = ap.parse_args(); torch.set_num_threads(1); pub = public_dir(a.data)
    M = D.load_machines(pub, "train"); lab = D.train_labels(pub); X = D.examples(M, lab)
    eps = sorted({m["ep"] for m in M}); fold = {e: i % 5 for i, e in enumerate(eps)}
    f = np.array([fold[e] for e in X["ep"]]); tr, ho = f != a.fold, f == a.fold
    sub = lambda m: {k: v[m] for k, v in X.items()}; Xtr, Xho = sub(tr), sub(ho)
    Xtr["W"] = shuffle_obs(Xtr["W"], cv.SEED); Xho["W"] = shuffle_obs(Xho["W"], cv.SEED + 1)
    t0 = time.time(); net = cv.train_gru(Xtr, "dense"); P = cv.predict_gru(net, Xho, "dense")
    # training-noise control: the UNshuffled dense GRU (S1) retrained with a different seed
    Xtr0, Xho0 = sub(tr), sub(ho); keep = cv.SEED; cv.SEED = 1
    P1 = cv.predict_gru(cv.train_gru(Xtr0, "dense"), Xho0, "dense"); cv.SEED = keep
    np.savez(f"method5_simulation/results/cv_ablate_fold{a.fold}.npz", S3b=P, S1_seed1=P1)
    print(f"ablate fold {a.fold}: S3b and S1(seed 1) trained in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
