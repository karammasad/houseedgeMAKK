"""Two-state hidden size regime per machine (standard library only). Owner: Person D.

Data finding (method4_ml/analysis/size_states.py): every validation machine's true size distribution given a
win is one of exactly TWO distributions per machine type, independent of signal and of the win state, and it
can switch over time. So: hidden state s in {0, 1}, emission e[type][s] over sizes (1, 2, 5, 20) on winning
ticks, losses and missing ticks carry no size information, and the state switches with probability h per tick.

fit(): Baum-Welch EM on pub/train histories (emissions per type, initial state prob, switch prob h).
predict(): forward filter over ticks < cutoff, one more transition to the cutoff tick, mix the two emissions.
"""
import math
from collections import defaultdict
from common.data import SIZES

IDX = {v: i for i, v in enumerate(SIZES)}


def _flip(h, k):
    """Probability the state differs after k ticks with per-tick switch prob h."""
    return (1 - (1 - 2 * h) ** k) / 2


def _obs(seq_m, cut):
    """(tick, size index) for every observed win before cut."""
    return [(t, IDX[v]) for t, v in enumerate(seq_m[:cut]) if v > 0]


class SizeHMM:
    def __init__(self, h=None, iters=40, h_init=0.003, pi_init=0.5):
        self.h_fixed, self.iters, self.h, self.pi0 = h, iters, (h if h is not None else h_init), pi_init

    # ---------- inference ----------
    def _forward(self, obs, E, cut):
        """Filtered P(state) at tick `cut` (before observing it)."""
        p = [1 - self.pi0, self.pi0]; last = 0
        for t, j in obs:
            f = _flip(self.h, t - last); p = [p[0] * (1 - f) + p[1] * f, p[1] * (1 - f) + p[0] * f]
            p = [p[0] * E[0][j], p[1] * E[1][j]]; s = p[0] + p[1]; p = [p[0] / s, p[1] / s]; last = t
        f = _flip(self.h, cut - last)
        return [p[0] * (1 - f) + p[1] * f, p[1] * (1 - f) + p[0] * f]

    def state_prob(self, q, seq_m):
        return self._forward(_obs(seq_m, q["cutoff"]), self.E[q["machine_type"]], q["cutoff"])

    def predict(self, q, seq_m):
        p = self.state_prob(q, seq_m); E = self.E[q["machine_type"]]
        return [p[0] * a + p[1] * b for a, b in zip(E[0], E[1])]

    # ---------- learning ----------
    def fit(self, queries, seq):
        data = defaultdict(list)
        for r in queries:
            o = _obs(seq[r["machine_id"]], r["cutoff"])
            if o:
                data[r["machine_type"]].append(o)
        # init: split machines by their 1x share
        self.E = {}
        for t, ms in data.items():
            share = sorted(sum(j == 0 for _, j in o) / len(o) for o in ms); med = share[len(share) // 2]
            acc = [[1.0] * 4, [1.0] * 4]
            for o in ms:
                k = int(sum(j == 0 for _, j in o) / len(o) < med)
                for _, j in o:
                    acc[k][j] += 1
            self.E[t] = [[x / sum(a) for x in a] for a in acc]
        for _ in range(self.iters):
            accE = {t: [[1e-3] * 4, [1e-3] * 4] for t in data}; acc_pi = [0.0, 0.0]; sw = 0.0; gaps = 0.0
            for t, ms in data.items():
                E = self.E[t]
                for o in ms:
                    g, xi = self._fb(o, E)
                    for (tt, j), gg in zip(o, g):
                        accE[t][0][j] += gg[0]; accE[t][1][j] += gg[1]
                    acc_pi[0] += g[0][0]; acc_pi[1] += g[0][1]
                    for k, (pf, gap) in enumerate(xi):  # expected flips per gap -> per-tick h via moment match
                        sw += pf; gaps += gap
            for t in data:
                self.E[t] = [[x / sum(a) for x in a] for a in accE[t]]
            self.pi0 = acc_pi[1] / sum(acc_pi)
            if self.h_fixed is None:
                self.h = self._solve_h(sw, gaps)
        return self

    def _solve_h(self, sw, gaps):
        # approximate M-step: expected flips per tick (small-h regime, flips over a gap ~ k*h)
        return min(max(sw / max(gaps, 1), 1e-6), 0.2)

    def _fb(self, o, E):
        n = len(o); h = self.h
        F = []; p = [1 - self.pi0, self.pi0]; last = o[0][0]; fl = [0.0] * n
        for i, (t, j) in enumerate(o):
            if i:
                f = _flip(h, t - last); fl[i] = f; p = [p[0] * (1 - f) + p[1] * f, p[1] * (1 - f) + p[0] * f]
            p = [p[0] * E[0][j], p[1] * E[1][j]]; s = p[0] + p[1]; p = [p[0] / s, p[1] / s]; F.append(p); last = t
        B = [[1.0, 1.0] for _ in range(n)]
        for i in range(n - 2, -1, -1):
            f = fl[i + 1]; j = o[i + 1][1]; b = B[i + 1]
            v0 = (1 - f) * E[0][j] * b[0] + f * E[1][j] * b[1]
            v1 = f * E[0][j] * b[0] + (1 - f) * E[1][j] * b[1]; s = v0 + v1; B[i] = [v0 / s, v1 / s]
        G = []
        for a, b in zip(F, B):
            x = [a[0] * b[0], a[1] * b[1]]; s = x[0] + x[1]; G.append([x[0] / s, x[1] / s])
        xi = []
        for i in range(1, n):
            f = fl[i]; j = o[i][1]; a = F[i - 1]; b = B[i]
            same = a[0] * (1 - f) * E[0][j] * b[0] + a[1] * (1 - f) * E[1][j] * b[1]
            diff = a[0] * f * E[1][j] * b[1] + a[1] * f * E[0][j] * b[0]
            xi.append((diff / (same + diff), o[i][0] - o[i - 1][0]))
        return G, xi
