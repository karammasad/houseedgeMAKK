"""Train the frozen Method 5 candidate ONCE on all permitted TRAIN data and export plain-JSON weights (offline, torch).

  python -m method5_simulation.train_final --mode dense          -> weights/s1.json (+ weights/s1.meta.json)

Training data: self-supervised late ticks of every TRAIN machine + the 768 labelled train cutoff queries. No validation or
test labels/targets are used. The configuration (hidden size, window, epochs, lr, batch, seed) is the one fixed in cv.py
before any result was seen; nothing is tuned here. Deterministic given the seed.
"""
import argparse, json, os, time
import numpy as np
import torch
from common.data import public_dir
from method5_simulation import data as D, cv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="dense"); ap.add_argument("--data", default="data/main")
    ap.add_argument("--out", default="method5_simulation/weights/s1.json"); a = ap.parse_args()
    torch.set_num_threads(1); pub = public_dir(a.data)
    M = D.load_machines(pub, "train"); X = D.examples(M, D.train_labels(pub)); t0 = time.time()
    net = cv.train_gru(X, a.mode); secs = time.time() - t0
    P = {k: v.detach().numpy().astype(np.float64).tolist() for k, v in net.state_dict().items()}
    P["H"] = cv.H; os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(P, open(a.out, "w"))
    json.dump(dict(mode=a.mode, H=cv.H, L=D.L, epochs=cv.EPOCHS, batch=cv.BATCH, lr=cv.LR, seed=cv.SEED, n_train=int(len(X["y"])),
                   train_seconds_one_cpu=round(secs, 1), torch=torch.__version__), open(a.out.replace(".json", ".meta.json"), "w"), indent=1)
    print(f"trained on {len(X['y'])} examples in {secs:.0f}s; exported {a.out} ({os.path.getsize(a.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
