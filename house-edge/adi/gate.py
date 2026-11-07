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


def _cell_mixture(nz, kz, zc, cp):
    """Asymmetric learned cell prior (C5). cp = (a_on, b_on, a_off, b_off): one gate cell is the 'on' cell
    (win rate ~ Beta(a_on, b_on)), the other the 'off' cell. Which one is on is unknown, so average both
    assignments 50/50 (keeps the model symmetric, so polarity flips still just swap cells).
    Returns (log marginal likelihood, P(win) for cell zc, or for an unknown cell if zc is None)."""
    a1, b1, a0, b0 = cp
    def part(c, a_, b_):
        return betaln(kz[c] + a_, nz[c] - kz[c] + b_) - betaln(a_, b_)
    lA = part(0, a1, b1) + part(1, a0, b0)  # cell 0 on
    lB = part(0, a0, b0) + part(1, a1, b1)  # cell 1 on
    m = max(lA, lB); wA = math.exp(lA - m); wB = math.exp(lB - m); s = wA + wB
    ll = m + math.log(s / 2)
    def rate(c, a_, b_):
        return (kz[c] + a_) / (nz[c] + a_ + b_)
    if zc is None:  # gate input missing at the cutoff: average the two cells by their share of observed ticks
        n = nz[0] + nz[1]; f0 = (nz[0] + 1) / (n + 2)
        pA = f0 * rate(0, a1, b1) + (1 - f0) * rate(1, a0, b0)
        pB = f0 * rate(0, a0, b0) + (1 - f0) * rate(1, a1, b1)
    else:
        on_A = zc == 0  # under assignment A, cell 0 is on
        pA = rate(zc, a1, b1) if on_A else rate(zc, a0, b0)
        pB = rate(zc, a0, b0) if on_A else rate(zc, a1, b1)
    return ll, (wA * pA + wB * pB) / s


def machine_tables(seq_m, cut, taus, ST, a, b, cellprior=None):
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
            zc = None if xc is None else (xc if tau is None else xc ^ 1)
            if cellprior is None:
                ll = betaln(kz[0] + a, nz[0] - kz[0] + b) + betaln(kz[1] + a, nz[1] - kz[1] + b) - base
                if zc is None:
                    pr = (kz[0] + kz[1] + a) / (nz[0] + nz[1] + a + b)
                else:
                    pr = (kz[zc] + a) / (nz[zc] + a + b)
            else:
                ll, pr = _cell_mixture(nz, kz, zc, cellprior)
            rows.append((ll, pr))
        out.append(rows)
    v = [x for x in w if x >= 0]; n0, k0 = len(v), sum(v)
    out.append([(betaln(k0 + a, n0 - k0 + b) - betaln(a, b), (k0 + a) / (n0 + a + b))] * (len(taus) + 1))
    return out


def learn_structure_prior(queries, seq, ST, a=.5, b=.5, p_change=.5, step=4, tau_min=40, cellprior=None):
    """Empirical-Bayes prior over structures: mean per-machine posterior on train, marginalizing the flip time
    (unlike method 1, which ignores flips here and so under-credits the true gate on drift/shared machines)."""
    acc = [0.0] * (len(ST) + 1)
    for r in queries:
        cut = r["cutoff"]; taus = list(range(tau_min, cut - 2, step))
        lp = [math.log(p_change / len(taus))] * len(taus) + [math.log(1 - p_change)]
        tb = machine_tables(seq[r["machine_id"]], cut, taus, ST, a, b, cellprior)
        l = [logsumexp([ll + p for (ll, _), p in zip(rows, lp)]) for rows in tb]; z = logsumexp(l)
        for i, x in enumerate(l):
            acc[i] += math.exp(x - z)
    s = sum(acc)
    return [x / s for x in acc]


def win_probability(q, seq_m, ST, sprior, a=.5, b=.5, p_change=.5, step=2, tau_min=40, cellprior=None,
                    evidence=False):
    """Posterior-mean P(win at cutoff). With evidence=True returns (P(win), log marginal likelihood of the log)."""
    cut = q["cutoff"]; taus = list(range(tau_min, cut - 2, step))
    lprior = [math.log(p_change / len(taus))] * len(taus) + [math.log(1 - p_change)]
    tb = machine_tables(seq_m, cut, taus, ST, a, b, cellprior); J = []; PR = []
    for si, rows in enumerate(tb):
        if sprior[si] <= 0:
            continue
        ls = math.log(sprior[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lprior[ti]); PR.append(pr)
    z = logsumexp(J)
    p = sum(math.exp(j - z) * p for j, p in zip(J, PR))
    return (p, z) if evidence else p


MEANS_ON = (.3, .4, .5, .6, .7, .8, .85, .9, .95, .98)
MEANS_OFF = (.005, .01, .02, .04, .07, .1, .15, .2, .3)
STRENGTHS = (3, 6, 12, 25, 50, 100)


def fit_cell_prior(queries, seq, ST, sprior, cfg, step=4, rounds=2):
    """Empirical Bayes (C5): pick (a_on, b_on, a_off, b_off) maximizing the summed log marginal likelihood of the
    given train logs under the full model (structures x flip times), by coordinate ascent over
    (on mean, off mean, on strength, off strength) grids. Train only. Returns (cellprior, best log-evidence)."""
    kw = dict(cfg); kw["step"] = max(step, cfg.get("step", 2))
    def ev(x):
        mo, mf, ko, kf = x
        cp = (mo * ko, (1 - mo) * ko, mf * kf, (1 - mf) * kf)
        return sum(win_probability(r, seq[r["machine_id"]], ST, sprior, cellprior=cp, evidence=True, **kw)[1]
                   for r in queries)
    x = [.85, .05, 12, 12]; cur = ev(x); grids = (MEANS_ON, MEANS_OFF, STRENGTHS, STRENGTHS)
    for _ in range(rounds):
        for i, grid in enumerate(grids):
            for v in grid:
                if v == x[i]:
                    continue
                y = x[:]; y[i] = v; e = ev(y)
                if e > cur:
                    x, cur = y, e
    mo, mf, ko, kf = x
    return (mo * ko, (1 - mo) * ko, mf * kf, (1 - mf) * kf), cur


def fit_cell_priors(queries, seq, ST, sprior, cfg, group="type_signal", min_group=40):
    """Cell priors per group of train machines ("none" = one shared prior, "type", "type_signal").
    Groups smaller than min_group fall back to the machine type's prior, then to the shared one."""
    keyf = {"none": lambda r: (), "type": lambda r: (r["machine_type"],),
            "type_signal": lambda r: (r["machine_type"], r["signal"])}[group]
    out = {(): fit_cell_prior(queries, seq, ST, sprior, cfg)[0]}
    groups = {}
    for r in queries:
        k = keyf(r)
        for j in range(1, len(k) + 1):
            groups.setdefault(k[:j], []).append(r)
    for k in sorted(groups, key=len):  # types before type x signal
        if len(groups[k]) >= min_group:
            out[k] = fit_cell_prior(groups[k], seq, ST, sprior, cfg)[0]
    return out


def lookup_cell_prior(priors, r, group):
    k = {"none": (), "type": (r["machine_type"],), "type_signal": (r["machine_type"], r["signal"])}[group]
    while k not in priors:
        k = k[:-1]
    return priors[k]
