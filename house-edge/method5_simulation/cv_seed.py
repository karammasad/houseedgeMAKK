"""S2 (observed-only GRU) retrained with a second seed on the same folds: is its advantage seed-robust?
  python -m method5_simulation.cv_seed --fold k   -> results/cv_seed_fold{k}.npz  (key S2_seed1)"""
import argparse, time
import numpy as np, torch
from common.data import public_dir
from method5_simulation import data as D, cv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--data", default="data/main")
    a = ap.parse_args(); torch.set_num_threads(1); pub = public_dir(a.data)
    M = D.load_machines(pub, "train"); X = D.examples(M, D.train_labels(pub))
    eps = sorted({m["ep"] for m in M}); fold = {e: i % 5 for i, e in enumerate(eps)}
    f = np.array([fold[e] for e in X["ep"]]); tr, ho = f != a.fold, f == a.fold
    sub = lambda m: {k: v[m] for k, v in X.items()}; cv.SEED = 1; t0 = time.time()
    P = cv.predict_gru(cv.train_gru(sub(tr), "nogap"), sub(ho), "nogap")
    np.savez(f"method5_simulation/results/cv_seed_fold{a.fold}.npz", S2_seed1=P)
    print(f"S2 seed 1 fold {a.fold}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
