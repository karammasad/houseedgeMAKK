"""Payout-size models given a win (standard library only). Owner: Person B.

A size model maps a machine's history to a distribution over (1x, 2x, 5x, 20x) given a win.
Interface:  SizeModel.fit(train_queries, train_seq) -> self ;  SizeModel.predict(query, seq_m) -> [s1, s2, s5, s20]
"""
import math
from collections import defaultdict
from common.data import SIZES


def size_counts(seq_m):
    return [sum(1 for x in seq_m if x == v) for v in SIZES]


class DirichletSizes:
    """Shrink each machine's counts toward its type x signal average (strength = pseudo-counts)."""
    def __init__(self, strength=30.0):
        self.strength = strength

    def fit(self, queries, seq):
        d = defaultdict(lambda: [0, 0, 0, 0])
        for r in queries:
            c = d[(r["machine_type"], r["signal"])]
            for i, x in enumerate(size_counts(seq[r["machine_id"]])):
                c[i] += x
        self.prior = {k: [(x + 1) / (sum(v) + 4) for x in v] for k, v in d.items()}
        return self

    def predict(self, q, seq_m):
        pri = self.prior[(q["machine_type"], q["signal"])]; c = size_counts(seq_m); n = sum(c)
        return [(x + self.strength * p) / (n + self.strength) for x, p in zip(c, pri)]


def _digamma(x):
    r = 0.0
    while x < 6:
        r -= 1 / x; x += 1
    f = 1 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


class MinkaSizes:
    """B6 control: Dirichlet-multinomial per machine type, alpha fitted by Minka's fixed-point iteration
    on train machines' size counts (strength and mean both learned). Posterior mean = (c + alpha)/(n + sum alpha)."""
    def __init__(self, iters=500):
        self.iters = iters

    def fit(self, queries, seq):
        C = defaultdict(list)
        for r in queries:
            c = size_counts(seq[r["machine_id"]])
            if sum(c) > 0:
                C[r["machine_type"]].append(c)
        self.alpha = {}
        for t, cs in C.items():
            tot = [sum(c[i] for c in cs) for i in range(4)]; N = sum(tot)
            a = [10 * (x + 1) / (N + 4) for x in tot]
            for _ in range(self.iters):
                A = sum(a)
                den = sum(_digamma(sum(c) + A) - _digamma(A) for c in cs)
                a = [max(1e-6, a[i] * sum(_digamma(c[i] + a[i]) - _digamma(a[i]) for c in cs) / den) for i in range(4)]
            self.alpha[t] = a
        return self

    def predict(self, q, seq_m):
        a = self.alpha[q["machine_type"]]; c = size_counts(seq_m); s = sum(c) + sum(a)
        return [(x + y) / s for x, y in zip(c, a)]


class LagSizes:
    """Size given a win, conditioned on whether earlier spins won (B1 finding: per-machine, direction varies).

    For each lag k in `lags`, the covariate is win/loss at tick (cutoff - k). Counts of this machine's sizes in
    past ticks with the same covariate pattern are shrunk toward `base`'s prediction for the machine:
        p = (counts_same_cell + strength * base) / (n_cell + strength)
    With two lags the cells nest: the joint cell shrinks toward the lag-1-only cell, which shrinks toward base.
    `half_life` (ticks) optionally down-weights old wins. Missing ticks never form a cell.
    """
    def __init__(self, base, lags=(1,), strength=20.0, half_life=None):
        self.base, self.lags, self.strength, self.half_life = base, tuple(lags), strength, half_life

    def fit(self, queries, seq):
        self.base.fit(queries, seq)
        return self

    def _cell_probs(self, seq_m, lags, prior):
        cut = len(seq_m)
        cur = [seq_m[cut - k] for k in lags]
        if any(v < 0 for v in cur):
            return prior
        want = tuple(int(v > 0) for v in cur)
        c = [0.0] * 4
        for t in range(max(lags), cut):
            v = seq_m[t]
            if v <= 0:
                continue
            past = [seq_m[t - k] for k in lags]
            if any(x < 0 for x in past) or tuple(int(x > 0) for x in past) != want:
                continue
            w = 1.0 if self.half_life is None else 2.0 ** (-(cut - 1 - t) / self.half_life)
            c[SIZES.index(v)] += w
        n = sum(c)
        return [(x + self.strength * p) / (n + self.strength) for x, p in zip(c, prior)]

    def predict(self, q, seq_m):
        p = self.base.predict(q, seq_m)
        for i in range(1, len(self.lags) + 1):
            p = self._cell_probs(seq_m, self.lags[:i], p)
        return p


class NPMLESizes:
    """Kiefer-Wolfowitz NPMLE prior over machine size distributions, per machine type (Koenker & Gu).
    Atoms: smoothed size distributions of train machines with >= min_wins wins. Weights: EM on train.
    Prediction: posterior mean over atoms given the machine's own size counts."""
    def __init__(self, min_wins=30, iters=300, half_life=None):
        self.min_wins, self.iters, self.half_life = min_wins, iters, half_life

    def fit(self, queries, seq):
        atoms = defaultdict(list); C = defaultdict(list)
        for r in queries:
            c = size_counts(seq[r["machine_id"]]); C[r["machine_type"]].append(c)
            if sum(c) >= self.min_wins:
                atoms[r["machine_type"]].append([(x + .5) / (sum(c) + 2) for x in c])
        self.model = {}
        for t, A in atoms.items():
            logA = [[math.log(v) for v in a] for a in A]
            E = []
            for c in C[t]:
                row = [sum(c[i] * la[i] for i in range(4)) for la in logA]; m = max(row)
                E.append([math.exp(x - m) for x in row])
            w = [1 / len(A)] * len(A)
            for _ in range(self.iters):
                acc = [0.0] * len(A)
                for e in E:
                    pe = [x * y for x, y in zip(e, w)]; s = sum(pe)
                    for j, v in enumerate(pe):
                        acc[j] += v / s
                w = [a / len(E) for a in acc]
            self.model[t] = (A, logA, w)
        return self

    def predict(self, q, seq_m):
        A, logA, wt = self.model[q["machine_type"]]
        if self.half_life is None:
            c = size_counts(seq_m)
        else:  # B4: recency-weighted counts in the posterior (atoms/weights still fit on full train counts)
            cut = len(seq_m); c = [0.0] * 4
            for t, v in enumerate(seq_m):
                if v > 0:
                    c[SIZES.index(v)] += 2.0 ** (-(cut - 1 - t) / self.half_life)
        l = [sum(c[i] * la[i] for i in range(4)) + (math.log(x) if x > 0 else -1e300) for la, x in zip(logA, wt)]
        m = max(l); p = [math.exp(x - m) for x in l]; s = sum(p)
        return [sum(p[j] * A[j][i] for j in range(len(A))) / s for i in range(4)]
