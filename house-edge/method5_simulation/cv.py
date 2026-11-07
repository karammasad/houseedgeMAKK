"""Episode-grouped cross-validation of Method 5 candidates (offline research; torch + sklearn allowed here).

  python -m method5_simulation.cv --fold 0   (run folds 0..4; each writes results/cv_fold{k}.npz)

Folds group whole TRAIN episodes (sorted episode id, i mod 5). Training examples: self-supervised late ticks of the
training-fold machines + their labelled cutoff queries. Held-out: the other episodes' late-tick examples and their
labelled cutoff queries. No validation/test data, labels or targets are used. Strict, fixed budget: no tuning.

Models: B0 type x signal prior | B1 gradient boosting on order-agnostic window frequencies | B2 B1 + last 8 outcomes |
        S1 GRU, dense grid (observation flags carry gaps) | S2 GRU, observed ticks only (no gap information) |
        S3 GRU, window order shuffled (no order information)
"""
import argparse, os, time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from common.data import public_dir
from method5_simulation import data as D

H, EPOCHS, BATCH, LR, SEED = 32, 10, 512, 3e-3, 0


def window_freqs(W):
    out = []
    for w in (8, 16, 32, 64):
        x = W[:, -w:]; obs = (x >= 0).sum(1)
        out.append(np.stack([(x == k).sum(1) / np.maximum(obs, 1) for k in range(5)], 1)); out.append((obs / w)[:, None])
    return np.concatenate(out, 1)


def tab(X, with_last8):
    f = [X["S"], window_freqs(X["W"])]
    if with_last8:
        f.append(X["W"][:, -8:][:, ::-1].astype(np.float32))
    return np.concatenate(f, 1)


def seq_input(W, mode, seed):
    N = len(W)
    if mode == "nogap":      # keep order, drop unobserved ticks (pushed to the front as padding): gap information removed
        order = np.argsort(W >= 0, axis=1, kind="stable"); W = np.take_along_axis(W, order, 1)
    elif mode == "shuffle":  # destroy order, keep the multiset of (category, observed) in the window
        perm = np.random.default_rng(seed).permuted(np.tile(np.arange(D.L), (N, 1)), axis=1); W = np.take_along_axis(W, perm, 1)
    x = np.zeros((N, D.L, 6), dtype=np.float32)
    x[..., :5] = (W[..., None] == np.arange(5)); x[..., 5] = W >= 0
    return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(6, H, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(H + D.NS, 64), torch.nn.ReLU(), torch.nn.Linear(64, 5))

    def forward(self, x, s):
        _, h = self.gru(x); return torch.softmax(self.head(torch.cat([h[-1], s], 1)), 1)


def train_gru(Xtr, mode):
    torch.manual_seed(SEED); np.random.seed(SEED)
    x = torch.from_numpy(seq_input(Xtr["W"], mode, SEED)); s = torch.from_numpy(Xtr["S"])
    y = torch.nn.functional.one_hot(torch.from_numpy(Xtr["y"]), 5).float()
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=LR); g = torch.Generator().manual_seed(SEED); n = len(x)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]; p = net(x[b], s[b]); loss = ((p - y[b]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def predict_gru(net, X, mode):
    with torch.no_grad():
        return net(torch.from_numpy(seq_input(X["W"], mode, SEED + 1)), torch.from_numpy(X["S"])).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--data", default="data/main")
    a = ap.parse_args(); torch.set_num_threads(1); pub = public_dir(a.data)
    M = D.load_machines(pub, "train"); lab = D.train_labels(pub); X = D.examples(M, lab)
    eps = sorted({m["ep"] for m in M}); fold = {e: i % 5 for i, e in enumerate(eps)}
    f = np.array([fold[e] for e in X["ep"]]); tr, ho = f != a.fold, f == a.fold
    sub = lambda m: {k: v[m] for k, v in X.items()}; Xtr, Xho = sub(tr), sub(ho)
    out = dict(y=Xho["y"], ep=Xho["ep"], short=Xho["short"], is_query=Xho["is_query"], typ=np.array([M[i]["typ"] for i in Xho["mi"]]))
    t0 = time.time(); times = {}
    # B0 type x signal prior
    key = lambda S: (S[:, :3].argmax(1) * 3 + (S[:, 3] + 1)).astype(int)
    prior = np.zeros((9, 5)); cnt = np.zeros(9)
    for k, y in zip(key(Xtr["S"]), Xtr["y"]):
        prior[k, y] += 1; cnt[k] += 1
    prior = (prior + 1) / (cnt[:, None] + 5); out["B0"] = prior[key(Xho["S"])]
    for name, l8 in (("B1", False), ("B2", True)):
        t1 = time.time(); clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                                               l2_regularization=1.0, random_state=SEED).fit(tab(Xtr, l8), Xtr["y"])
        out[name] = clf.predict_proba(tab(Xho, l8)); times[name] = time.time() - t1
    for name, mode in (("S1", "dense"), ("S2", "nogap"), ("S3", "shuffle")):
        t1 = time.time(); net = train_gru(Xtr, mode); out[name] = predict_gru(net, Xho, mode); times[name] = time.time() - t1
    os.makedirs("method5_simulation/results", exist_ok=True)
    np.savez(f"method5_simulation/results/cv_fold{a.fold}.npz", **out)
    print(f"fold {a.fold}: train {int(tr.sum())} / held-out {int(ho.sum())} examples; seconds {({k: round(v) for k, v in times.items()})}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
