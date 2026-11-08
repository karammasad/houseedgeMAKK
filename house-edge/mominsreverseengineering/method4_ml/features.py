"""Leakage-safe per-tick features for ML models. Owner: Person D. Standard library only.

features_at(w, payouts, t, state) uses only ticks < t. Iterating t upward and calling update(state, w, t)
after each tick gives every training example its features without ever reading the label tick.
Per structure s (all single lags and lag pairs in 1..8):
  x_s       XOR of past wins at the lags in s (-1 if any is missing)
  agree_s   smoothed share of earlier ticks where win == x_s   (full history)
  recent_s  same share over the last RECENT ticks              (tracks polarity flips)
Plus: lag wins 1..12, lag payouts 1..12, overall win rate, recent win rate, ticks observed.
"""
import itertools as it
from collections import deque

LAGS = 12
STRUCTS = [(k,) for k in range(1, 9)] + list(it.combinations(range(1, 9), 2))
RECENT = 40


def new_state():
    return {"agree": [[0, 0] for _ in STRUCTS], "recent": [deque(maxlen=RECENT) for _ in STRUCTS],
            "wins": 0, "n": 0, "recent_w": deque(maxlen=RECENT)}


def _x(w, t, s):
    x = 0
    for l in s:
        if t - l < 0 or w[t - l] < 0:
            return -1
        x ^= w[t - l]
    return x


def features_at(w, payouts, t, state):
    f = [w[t - l] if t - l >= 0 else -1 for l in range(1, LAGS + 1)]
    f += [payouts[t - l] if t - l >= 0 else -1 for l in range(1, LAGS + 1)]
    for i, s in enumerate(STRUCTS):
        a, n = state["agree"][i]; rq = state["recent"][i]
        f += [_x(w, t, s), (a + 1) / (n + 2), (sum(rq) + 1) / (len(rq) + 2)]
    rw = state["recent_w"]
    f += [(state["wins"] + 1) / (state["n"] + 2), (sum(rw) + 1) / (len(rw) + 2), state["n"]]
    return f


def update(state, w, t):
    if w[t] < 0:
        return
    for i, s in enumerate(STRUCTS):
        x = _x(w, t, s)
        if x >= 0:
            hit = int(w[t] == x); state["agree"][i][0] += hit; state["agree"][i][1] += 1; state["recent"][i].append(hit)
    state["wins"] += w[t]; state["n"] += 1; state["recent_w"].append(w[t])


def machine_examples(seq_m, extra):
    """All (features, label) pairs from one machine's history; label = payout class index at tick t."""
    from common.data import PAYOUTS, wins
    w = wins(seq_m); st = new_state(); X, y = [], []
    for t in range(len(w)):
        if w[t] >= 0 and t >= LAGS:
            X.append(features_at(w, seq_m, t, st) + extra); y.append(PAYOUTS.index(seq_m[t]))
        update(st, w, t)
    return X, y, st


def query_features(seq_m, cut, extra):
    from common.data import wins
    w = wins(seq_m); st = new_state()
    for t in range(cut):
        update(st, w, t)
    return features_at(w + [-1], seq_m + [-1], cut, st) + extra
