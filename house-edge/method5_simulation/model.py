"""Method 5 shipping model: standard-library-only inference for the GRU sequence representation learner.

Interface (AGENTS.md): Model(**settings).fit(pub) then Model.predict(pub, split) -> {query_id: [p0, p1, p2, p5, p20]}.
Training is OFFLINE (train_final.py, torch, fixed seeds) on real TRAIN histories + train labels only; the exported
weights (weights/s1.json) are plain JSON. `fit` loads them (no network, no numpy/torch). For a new round, re-run
train_final.py on the new train folder, then this file reproduces the CSV.

Features for the query at tick t = cutoff use ONLY seq[:t] (the history, strictly earlier ticks), the machine type and
signal; missing ticks (-1) are flagged as unobserved and are never treated as losses. Imports nothing from Methods 1-4.
"""
import json, math, os
from operator import mul
from common.data import load_split

PAY = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
TYPES = ("steady", "balanced", "jackpot")
L = 64
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "s1.json")


def featurize(seq_m, t, typ, sig):
    """(window categories of ticks t-L..t-1 with -1 = unobserved, statics[17]); identical to data.build()."""
    cats = [-1 if v < 0 else PAY[v] for v in seq_m[:t]]
    window = [cats[u] if u >= 0 else -1 for u in range(t - L, t)]
    pre = [0] * 5
    for c in cats:
        if c >= 0:
            pre[c] += 1
    npre = sum(pre); rec = [0] * 5
    for c in cats[max(t - 32, 0):t]:
        if c >= 0:
            rec[c] += 1
    nrec = sum(rec)
    S = [1.0 if typ == k else 0.0 for k in TYPES] + [float(sig), t / 256.0]
    S += [pre[k] / max(npre, 1) for k in range(5)] + [math.log1p(npre) / 6.0]
    S += [rec[k] / max(nrec, 1) for k in range(5)] + [nrec / 32.0]
    return window, S


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -60 else 0.0


class Model:
    def __init__(self, weights=WEIGHTS, mode=None):
        self.path, self.mode = weights, mode

    def fit(self, pub=None):
        with open(self.path) as f:
            P = json.load(f)
        if self.mode is None:   # input mode is recorded by train_final.py next to the weights
            with open(self.path.replace(".json", ".meta.json")) as mf:
                self.mode = json.load(mf)["mode"]
        self.H = P["H"]; H = self.H
        wih, whh = P["gru.weight_ih_l0"], P["gru.weight_hh_l0"]        # (3H, 6), (3H, H); gate order r, z, n
        self.cols = [[wih[j][c] for j in range(3 * H)] for c in range(6)]  # input is one-hot(5) + observed flag: sparse
        self.bih, self.bhh, self.whh = P["gru.bias_ih_l0"], P["gru.bias_hh_l0"], whh
        self.w1, self.b1, self.w2, self.b2 = P["head.0.weight"], P["head.0.bias"], P["head.2.weight"], P["head.2.bias"]
        return self

    def forward(self, window, S):
        H = self.H; h = [0.0] * H; bih, bhh, whh, cols = self.bih, self.bhh, self.whh, self.cols
        for c in window:
            gi = bih[:]                                   # input projection: add the category column and the flag column
            if c >= 0:
                a, b = cols[c], cols[5]
                gi = [g + x + y for g, x, y in zip(gi, a, b)]
            gh = [sum(map(mul, row, h)) + b for row, b in zip(whh, bhh)]
            r = [_sig(gi[j] + gh[j]) for j in range(H)]
            z = [_sig(gi[H + j] + gh[H + j]) for j in range(H)]
            n = [math.tanh(gi[2 * H + j] + r[j] * gh[2 * H + j]) for j in range(H)]
            h = [(1.0 - z[j]) * n[j] + z[j] * h[j] for j in range(H)]
        v = h + S
        a1 = [max(0.0, sum(map(mul, row, v)) + b) for row, b in zip(self.w1, self.b1)]
        lo = [sum(map(mul, row, a1)) + b for row, b in zip(self.w2, self.b2)]
        m = max(lo); e = [math.exp(x - m) for x in lo]; s = sum(e)
        return [x / s for x in e]

    def predict(self, pub, split):
        q, seq = load_split(pub, split); out = {}
        for r in q:
            window, S = featurize(seq[r["machine_id"]], r["cutoff"], r["machine_type"], int(r["signal"]))
            if self.mode == "nogap":   # observed ticks only, order kept, unobserved pushed to the front as padding
                obs = [c for c in window if c >= 0]; window = [-1] * (L - len(obs)) + obs
            out[r["query_id"]] = self.forward(window, S)
        return out
