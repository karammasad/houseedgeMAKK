"""Bayesian change-point gate model (standard library only). Owner: Person A.

P(win at T) = sum over structures s and change times tau of P(s, tau | log) * (k_z + a) / (n_z + a + b),
where z = XOR of past wins at the lags in s, flipped after tau. Each gate cell is a Beta(a, b)-Bernoulli rate.
Structures: every single lag and lag pair in 1..max_lag, plus 'no gate'. Prefix sums make each tau O(1).
"""
import itertools as it, math
from bisect import bisect_left
from collections import defaultdict
from common.data import wins


def betaln(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def logsumexp(v):
    m = max(v); return m + math.log(sum(math.exp(x - m) for x in v))


def structures(max_lag=8, max_order=2):
    S = []
    for k in range(1, max_order + 1):
        S += list(it.combinations(range(1, max_lag + 1), k))
    return S


def machine_tables(seq_m, cut, taus, ST, a, b):
    """For each structure (plus 'no gate' last): list over taus (+ None = no flip) of (log marginal likelihood, P(win))."""
    w = wins(seq_m); out = []; base = 2 * betaln(a, b)
    for st in ST:
        ts, xs, ys = [], [], []
        for t in range(max(st), len(w)):
            if w[t] < 0 or any(w[t - l] < 0 for l in st):
                continue
            x = 0
            for l in st:
                x ^= w[t - l]
            ts.append(t); xs.append(x); ys.append(w[t])
        n = [[0], [0]]; k = [[0], [0]]
        for x, y in zip(xs, ys):
            for c in (0, 1):
                n[c].append(n[c][-1] + (x == c)); k[c].append(k[c][-1] + (x == c and y == 1))
        N = [n[0][-1], n[1][-1]]; K = [k[0][-1], k[1][-1]]
        xc = None
        if all(w[cut - l] >= 0 for l in st):
            xc = 0
            for l in st:
                xc ^= w[cut - l]
        rows = []
        for tau in list(taus) + [None]:
            if tau is None:
                nz, kz = N[:], K[:]
            else:
                i = bisect_left(ts, tau)  # ticks before tau keep cell x; from tau on the cell flips
                nz = [n[0][i] + (N[1] - n[1][i]), n[1][i] + (N[0] - n[0][i])]
                kz = [k[0][i] + (K[1] - k[1][i]), k[1][i] + (K[0] - k[0][i])]
            ll = betaln(kz[0] + a, nz[0] - kz[0] + b) + betaln(kz[1] + a, nz[1] - kz[1] + b) - base
            if xc is None:
                pr = (kz[0] + kz[1] + a) / (nz[0] + nz[1] + a + b)
            else:
                zc = xc if tau is None else xc ^ 1
                pr = (kz[zc] + a) / (nz[zc] + a + b)
            rows.append((ll, pr))
        out.append(rows)
    v = [x for x in w if x >= 0]; n0, k0 = len(v), sum(v)
    out.append([(betaln(k0 + a, n0 - k0 + b) - betaln(a, b), (k0 + a) / (n0 + a + b))] * (len(taus) + 1))
    return out


def learn_structure_prior(queries, seq, ST, a=.5, b=.5, none_floor=.05, uniform_mix=.1):
    """Empirical-Bayes structure prior: mean per-machine posterior over structures (no change point)."""
    acc = [0.0] * (len(ST) + 1)
    for r in queries:
        tb = machine_tables(seq[r["machine_id"]], r["cutoff"], [], ST, a, b)
        l = [t[0][0] for t in tb]; z = logsumexp(l)
        for i, x in enumerate(l):
            acc[i] += math.exp(x - z)
    s = sum(acc); p = [(1 - uniform_mix) * x / s + uniform_mix / len(acc) for x in acc]
    p[-1] = max(p[-1], none_floor); s = sum(p)
    return [x / s for x in p]


def win_probability(q, seq_m, ST, sprior, a=.5, b=.5, p_change=.5, step=2, tau_min=40):
    cut = q["cutoff"]; taus = list(range(tau_min, cut - 2, step))
    lprior = [math.log(p_change / len(taus))] * len(taus) + [math.log(1 - p_change)]
    tb = machine_tables(seq_m, cut, taus, ST, a, b); J = []; PR = []
    for si, rows in enumerate(tb):
        ls = math.log(sprior[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lprior[ti]); PR.append(pr)
    z = logsumexp(J)
    return sum(math.exp(j - z) * p for j, p in zip(J, PR))
