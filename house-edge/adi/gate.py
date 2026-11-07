"""Generalized change-point gate (adi / method 3). Standard library only.

Same Bayesian model as method1_gate.gate, but a structure is (form, lags) instead of just lags:
  form "xor"          : XOR of past wins at the lags (any order; ("xor", (l,)) is a single lag)
  form "and" + mask m : AND over lags of (win at lag) XOR (bit i of m), i.e. AND with optional input negations.
                        OR gates are negated ANDs with negated inputs, and the gate's two cells have free rates,
                        so the AND family with all 4 masks already covers OR / NAND / NOR.
Each structure's two gate cells are Beta(a, b)-Bernoulli rates; one optional polarity flip at tick tau swaps them.
The model averages over structures (with a learned prior) and flip times, exactly as method 1 does.
"""
import itertools as it, math
from bisect import bisect_left
from common.data import wins
from method1_gate.gate import betaln, logsumexp


def structures(max_lag=8, xor_order=2, and_pairs=False, and_max_lag=None):
    S = [("xor", s) for k in range(1, xor_order + 1) for s in it.combinations(range(1, max_lag + 1), k)]
    if and_pairs:
        L = and_max_lag or max_lag
        for s in it.combinations(range(1, L + 1), 2):
            for m in range(4):
                S.append((f"and{m}", s))
    return S


def gate_value(form, lags, w, t):
    """Gate bit at tick t from wins strictly before t, or None if any input tick is missing."""
    if form == "xor":
        x = 0
        for l in lags:
            v = w[t - l]
            if v < 0:
                return None
            x ^= v
        return x
    m = int(form[3:]); x = 1
    for i, l in enumerate(lags):
        v = w[t - l]
        if v < 0:
            return None
        x &= v ^ ((m >> i) & 1)
    return x


def machine_tables(seq_m, cut, taus, ST, a, b):
    """Per structure (plus 'no gate' last): list over taus (+ None = no flip) of (log marginal lik, P(win at cut))."""
    w = wins(seq_m); out = []; base = 2 * betaln(a, b)
    for form, st in ST:
        ts, xs, ys = [], [], []
        for t in range(max(st), len(w)):
            if w[t] < 0:
                continue
            x = gate_value(form, st, w, t)
            if x is None:
                continue
            ts.append(t); xs.append(x); ys.append(w[t])
        n = [[0], [0]]; k = [[0], [0]]
        for x, y in zip(xs, ys):
            for c in (0, 1):
                n[c].append(n[c][-1] + (x == c)); k[c].append(k[c][-1] + (x == c and y == 1))
        N = [n[0][-1], n[1][-1]]; K = [k[0][-1], k[1][-1]]
        xc = gate_value(form, st, w, cut)
        rows = []
        for tau in list(taus) + [None]:
            if tau is None:
                nz, kz = N[:], K[:]
            else:
                i = bisect_left(ts, tau)
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


def learn_structure_prior(queries, seq, ST, a=.5, b=.5, p_change=.5, step=4, tau_min=40):
    """Empirical-Bayes prior over structures: mean per-machine posterior on train, marginalizing the flip time
    (unlike method 1, which ignores flips here and so under-credits the true gate on drift/shared machines)."""
    acc = [0.0] * (len(ST) + 1)
    for r in queries:
        cut = r["cutoff"]; taus = list(range(tau_min, cut - 2, step))
        lp = [math.log(p_change / len(taus))] * len(taus) + [math.log(1 - p_change)]
        tb = machine_tables(seq[r["machine_id"]], cut, taus, ST, a, b)
        l = [logsumexp([ll + p for (ll, _), p in zip(rows, lp)]) for rows in tb]; z = logsumexp(l)
        for i, x in enumerate(l):
            acc[i] += math.exp(x - z)
    s = sum(acc)
    return [x / s for x in acc]


def win_probability(q, seq_m, ST, sprior, a=.5, b=.5, p_change=.5, step=2, tau_min=40):
    cut = q["cutoff"]; taus = list(range(tau_min, cut - 2, step))
    lprior = [math.log(p_change / len(taus))] * len(taus) + [math.log(1 - p_change)]
    tb = machine_tables(seq_m, cut, taus, ST, a, b); J = []; PR = []
    for si, rows in enumerate(tb):
        if sprior[si] <= 0:
            continue
        ls = math.log(sprior[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lprior[ti]); PR.append(pr)
    z = logsumexp(J)
    return sum(math.exp(j - z) * p for j, p in zip(J, PR))
