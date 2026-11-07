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


class NPMLESizes:
    """Kiefer-Wolfowitz NPMLE prior over machine size distributions, per machine type (Koenker & Gu).
    Atoms: smoothed size distributions of train machines with >= min_wins wins. Weights: EM on train.
    Prediction: posterior mean over atoms given the machine's own size counts."""
    def __init__(self, min_wins=30, iters=300):
        self.min_wins, self.iters = min_wins, iters

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
        A, logA, wt = self.model[q["machine_type"]]; c = size_counts(seq_m)
        l = [sum(c[i] * la[i] for i in range(4)) + (math.log(x) if x > 0 else -1e300) for la, x in zip(logA, wt)]
        m = max(l); p = [math.exp(x - m) for x in l]; s = sum(p)
        return [sum(p[j] * A[j][i] for j in range(len(A))) / s for i in range(4)]
