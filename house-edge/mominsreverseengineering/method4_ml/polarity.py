"""Polarity bank (Person D, v9): mechanism-agnostic, machine-specific evidence features for the stacker. numpy only.

Why: the win gate's structure family uses only the machine's OWN past wins (singles, XOR / AND pairs). The rules say other
machines' earlier outcomes "may help" and that round two changes "which temporal relationships recur"; trees cannot discover
parities or machine-specific polarities from raw lag inputs (TASK.md). So for each machine we test, on its own history, every
candidate bit b(u) from a bank, and hand the trees the strongest evidence plus the signed prediction it makes NOW:

  bank NB  win of neighbour machine j at tick u - l (j != m, l = 1..L)                       (cross-machine rules)
  bank SZ  own payout >= 2 at tick u - l (l = 1..L)                                          (size-category rules)
  bank NX  own win at u - l XOR neighbour j's win at u - l (same lag)                        (cross-machine parities)

For candidate c with bit b: win rates r1 = P(win | b = 1), r0 = P(win | b = 0) over ticks u < t where every input is observed
(Beta(.5, .5) smoothing); z_c = (r1 - r0) / se. Features per bank: |z| of the best and second-best candidate and, for each,
its predicted shift (r_b(now) - own rate). Under no real rule the max |z| over the bank follows a null distribution the trees
see in thousands of training rows, so they learn how much evidence is needed before trusting it (no hand-set threshold).
Leakage: counts use u < t only and the current bit uses ticks t - l < t; the label tick t is never an input.
"""
import numpy as np

L = 8
NFEAT = 12   # 3 banks x (z1, shift1, z2, shift2)


def _bank_stats(B, y, t):
    """B: (C, t) candidate bits in {-1 (missing), 0, 1} for ticks 0..t-1 aligned with y (t,) wins in {-1, 0, 1}; also B_now (C,)
    is passed as B[:, t] (column t). Returns (z (C,), shift (C,))."""
    Bp, bn = B[:, :t], B[:, t]
    ok = (Bp >= 0) & (y[None, :t] >= 0); win = (y[None, :t] == 1)
    one = ok & (Bp == 1); zer = ok & (Bp == 0)
    n1, n0 = one.sum(1), zer.sum(1); k1, k0 = (one & win).sum(1), (zer & win).sum(1)
    r1, r0 = (k1 + .5) / (n1 + 1), (k0 + .5) / (n0 + 1)
    se = np.sqrt(r1 * (1 - r1) / (n1 + 1) + r0 * (1 - r0) / (n0 + 1))
    z = (r1 - r0) / se
    yo = y[:t][y[:t] >= 0]; base = (yo.sum() + .5) / (len(yo) + 1)
    now = np.where(bn == 1, r1, np.where(bn == 0, r0, base))
    return z, now - base


def _top2(z, sh):
    if len(z) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    a = np.abs(z); o = np.argsort(-a, kind="stable")
    i = o[0]; j = o[1] if len(o) > 1 else o[0]
    return [float(a[i]), float(sh[i]), float(a[j]), float(sh[j])]


def _lagged(v, l, T):
    """v shifted by l: out[u] = v[u - l] (u = 0..T), -1 where u - l < 0. v has length >= T."""
    out = np.full(T + 1, -1, dtype=np.int8)
    if l <= T:
        out[l:] = v[:T + 1 - l]
    return out


def episode_polarity(W, P, t):
    """W: (M, >= t) wins {-1, 0, 1}; P: (M, >= t) payouts (-1 missing). Features predicting tick t for every machine. Returns
    (M, NFEAT). Only columns < t are read (column t of the lagged arrays holds bits from ticks t - l < t)."""
    M = W.shape[0]; Wt = np.asarray(W[:, :t], dtype=np.int8); Pt = np.asarray(P[:, :t])
    big = np.where(Pt < 0, -1, (Pt >= 2).astype(np.int8)).astype(np.int8)
    LW = np.stack([np.stack([_lagged(Wt[j], l, t) for l in range(1, L + 1)]) for j in range(M)])     # (M, L, t+1)
    LB = np.stack([np.stack([_lagged(big[j], l, t) for l in range(1, L + 1)]) for j in range(M)])
    miss = (LW[:, None] < 0) | (LW[None, :] < 0)                                                      # (M, M, L, t+1)
    NX = np.where(miss, -1, LW[:, None] ^ LW[None, :]).astype(np.int8)
    out = np.zeros((M, NFEAT))
    for m in range(M):
        y = Wt[m]; others = [j for j in range(M) if j != m]
        banks = (LW[others].reshape(-1, t + 1), LB[m], NX[m, others].reshape(-1, t + 1)) if others else (None, LB[m], None)
        f = []
        for bank in banks:
            if bank is not None and len(bank) and t > 0:
                z, sh = _bank_stats(bank, y, t); f += _top2(z, sh)
            else:
                f += [0.0] * 4
        out[m] = f
    return out
