"""Sparse two-lag conditional size expert (Method 2, experiment 8). Standard library only.

Idea (adapted from conditional tensor factorization, Sarkar & Dunson 2016, and Bayesian context trees /
context-tree weighting): the size state at tick t may depend on the payout categories at TWO lags (a, b) through a
low-dimensional latent map, and the structure should be averaged over, not selected.

Two-state factorization: each lag's payout category is collapsed to the learned group indicator
(big win vs not; the joint-EM partition prior is .96 on that split), giving a 2 x 2 context (g_a, g_b).
A map h: {0,1}^2 -> {lo, hi} assigns each context to one of the type's two learned P(1x|win) levels.
Only maps that genuinely depend on BOTH lags are used (XOR, XNOR and the 8 one-cell-different maps), so this
expert is a real alternative to the existing single-lag model, which stays as the fallback expert.

  P(sizes | pair (a,b), map h) = prod over wins: Bernoulli(is_1x; level[h(g_a, g_b)])
  wins with a missing lag: state unknown -> 50/50 mixture of the two levels (missing ticks never read as losses)
  pair prior:  "m1"      (1 - eps) * method 1's posterior mass on win pair (a,b) + eps * uniform over 28 pairs
               "m1_only" only pairs carrying >= 5% of method 1's posterior (fallback uniform if none)
               "uniform" uniform over all 28 pairs, independent of method 1
  map prior: uniform over the 10 two-lag maps.  Expert evidence = exact marginal over (pair, map).

mode="alone": use only this expert.  mode="mix": Bayesian mixture with the existing GateSizes expert;
the prior weight v of the new expert is learned by EM on train machines (posterior weight averaged).
The big-win split always comes from the base GateSizes model.
"""
import itertools, math
from method1_gate.gate import logsumexp

PAIRS = list(itertools.combinations(range(1, 9), 2))
CTX = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _maps():
    out = []
    for bits in itertools.product((0, 1), repeat=4):
        h = dict(zip(CTX, bits))
        dep_a = any(h[(0, y)] != h[(1, y)] for y in (0, 1)); dep_b = any(h[(x, 0)] != h[(x, 1)] for x in (0, 1))
        if dep_a and dep_b:
            out.append(h)
    return out


MAPS = _maps()   # 10 maps that depend on both lags


class TwoLagSizes:
    def __init__(self, base, mode="mix", pairs="m1", eps=0.2, v_init=0.1, iters=30):
        self.base, self.mode, self.pairs, self.eps, self.v, self.iters = base, mode, pairs, eps, v_init, iters

    # ---- the new expert -------------------------------------------------------------------------------------
    def _pair_prior(self, seq_m):
        if self.pairs == "uniform" or not self.base.couple:
            return [1 / len(PAIRS)] * len(PAIRS)
        wp = self.base._win_posterior(seq_m); mass = {p: 0.0 for p in PAIRS}
        for st, x in zip(list(self.base.wST) + [()], wp):
            if len(st) == 2:
                mass[tuple(st)] = mass.get(tuple(st), 0.0) + x
        if self.pairs == "m1_only":
            keep = {p: x for p, x in mass.items() if x >= .05}
            if not keep:
                return [1 / len(PAIRS)] * len(PAIRS)
            z = sum(keep.values()); return [keep.get(p, 0.0) / z for p in PAIRS]
        z = sum(mass.values())
        return [(1 - self.eps) * (mass[p] / z if z > 0 else 1 / len(PAIRS)) + self.eps / len(PAIRS) for p in PAIRS]

    def _expert(self, t, seq_m):
        """(log evidence, predicted P(1x|win)) of the two-lag expert, exact over (pair, map)."""
        lo, hi = self.base.levels[t]; lv = (math.log(lo), math.log(1 - lo)), (math.log(hi), math.log(1 - hi))
        mid = (lo + hi) / 2; lm = (math.log(mid), math.log(1 - mid))
        cut = len(seq_m); wins = [(u, int(seq_m[u] == 1)) for u in range(cut) if seq_m[u] > 0]
        pp = self._pair_prior(seq_m); terms = []
        for (a, b), pr in zip(PAIRS, pp):
            if pr <= 0:
                continue
            cnt = {c: [0, 0] for c in CTX}; miss = [0, 0]
            for u, y in wins:
                xa = seq_m[u - a] if u - a >= 0 else -1; xb = seq_m[u - b] if u - b >= 0 else -1
                if xa < 0 or xb < 0:
                    miss[y] += 1
                else:
                    cnt[(int(xa > 1), int(xb > 1))][y] += 1
            llm = miss[1] * lm[0] + miss[0] * lm[1]
            ca = seq_m[cut - a] if cut - a >= 0 else -1; cb = seq_m[cut - b] if cut - b >= 0 else -1
            cur = None if ca < 0 or cb < 0 else (int(ca > 1), int(cb > 1))
            lpm = math.log(pr / len(MAPS))
            for h in MAPS:
                ll = llm + sum(cnt[c][1] * lv[h[c]][0] + cnt[c][0] * lv[h[c]][1] for c in CTX)
                terms.append((lpm + ll, mid if cur is None else (lo, hi)[h[cur]]))
        z = logsumexp([x for x, _ in terms])
        return z, sum(math.exp(x - z) * aa for x, aa in terms)

    def _old(self, q, seq_m):
        t = q["machine_type"]; b = self.base
        tabs = b._tables(seq_m, b.mu[t], t); pr = b.size_prior(seq_m)
        z = logsumexp([math.log(max(p, 1e-300)) + ll for p, (ll, _) in zip(pr, tabs)])
        full = b.predict(q, seq_m)
        return z, full[0], [x / (1 - full[0]) for x in full[1:]] if full[0] < 1 else [1 / 3] * 3

    # ---- interface -------------------------------------------------------------------------------------------
    def fit(self, queries, seq):
        self.base.fit(queries, seq)
        self.trace = []
        if self.mode != "mix":
            return self
        rows = []
        for r in queries:
            t = r["machine_type"]
            if t in (self.base.levels or {}):
                s = seq[r["machine_id"]]
                rows.append((self._old(r, s)[0], self._expert(t, s)[0]))
        v = self.v
        for _ in range(self.iters):   # EM for the prior weight of the new expert
            acc = 0.0
            for lo_, ln in rows:
                x, y = math.log(1 - v) + lo_, math.log(v) + ln; m = max(x, y)
                acc += math.exp(y - m) / (math.exp(x - m) + math.exp(y - m))
            v = min(.99, max(.01, acc / max(1, len(rows)))); self.trace.append(v)
        self.v = v
        return self

    def weight_new(self, q, seq_m):
        t = q["machine_type"]
        if t not in (self.base.levels or {}):
            return 0.0
        if self.mode == "alone":
            return 1.0
        lo_, _, _ = self._old(q, seq_m); ln, _ = self._expert(t, seq_m)
        x, y = math.log(1 - self.v) + lo_, math.log(self.v) + ln; m = max(x, y)
        return math.exp(y - m) / (math.exp(x - m) + math.exp(y - m))

    def predict(self, q, seq_m):
        t = q["machine_type"]
        _, a_old, sp = self._old(q, seq_m)
        if t not in (self.base.levels or {}):
            a = a_old
        else:
            _, a_new = self._expert(t, seq_m); w = self.weight_new(q, seq_m)
            a = w * a_new + (1 - w) * a_old
        return [a] + [(1 - a) * x for x in sp]
