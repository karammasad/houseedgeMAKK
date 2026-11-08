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
    if form in ("xs", "xst"):  # size-split forms: the gate bit is the XOR of the pair (the split only refines cells)
        form, lags = "xor", lags[:2]
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
    a1, b1, a0, b0 = cp[:4]
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


def _cat3(v):
    return 0 if v == 0 else (1 if v == 1 else 2)  # loss / 1x / big (v >= 0 assumed)


def _split_flip(c):
    return (c + 3) % 6  # swap the XOR side, keep the size category


def _sh(a, b, d):
    """Shift the Beta mean by d on the log-odds scale, strength kept."""
    if d == 0.0:
        return a, b
    k = a + b; m = min(max(a / k, 1e-6), 1 - 1e-6)
    m = 1 / (1 + math.exp(-(math.log(m / (1 - m)) + d)))
    return m * k, (1 - m) * k


def _split_mixture(nz, kz, zc, cp, a=.5, b=.5):
    """Size-split gate ("xs"): 6 cells = XOR side (0/1) x payout category at the split lag (loss/1x/big).
    With cp = (a_on, b_on, a_off, b_off), the three sub-cells on the 'on' side use the on prior and the
    others the off prior, averaged 50/50 over which side is on (as in _cell_mixture). Each sub-cell has its
    own rate, so a machine whose win rate depends on its hidden size state can express it.
    STATE SHIFT (mominsreverseengineering): if cp has a 5th element gamma > 0, each sub-cell's prior mean is shifted by
    +-gamma on the log-odds scale: ONE hidden binary size state (big category vs the other two, direction d unknown,
    averaged 50/50) moves every sub-cell the same way on both sides. gamma = 0 reproduces the exchangeable prior exactly."""
    if cp is None:
        cp = (a, b, a, b)
    a1, b1, a0, b0 = cp[:4]; gam = cp[4] if len(cp) > 4 else 0.0
    comps = []
    for assign in (0, 1):                       # assign 0: side 0 is the 'on' side
        for d in ((0, 1) if gam else (0,)):
            ll_ = 0.0; pri = []
            for c in range(6):
                on = (c < 3) == (assign == 0)
                sig = 1 if ((c % 3 == 2) != (d == 1)) else -1
                pa, pb = _sh(*((a1, b1) if on else (a0, b0)), gam * sig)
                pri.append((pa, pb))
                if nz[c]:
                    ll_ += betaln(kz[c] + pa, nz[c] - kz[c] + pb) - betaln(pa, pb)
            comps.append((ll_, pri))
    m = max(l for l, _ in comps); ws = [math.exp(l - m) for l, _ in comps]; s = sum(ws)
    ll = m + math.log(s / len(comps))
    def rate(c, pri):
        pa, pb = pri[c]
        return (kz[c] + pa) / (nz[c] + pa + pb)
    if zc is None:
        n = sum(nz); f = [(nz[c] + 1 / 6) / (n + 1) for c in range(6)]
        ps = [sum(f[c] * rate(c, pri) for c in range(6)) for _, pri in comps]
    else:
        ps = [rate(zc, pri) for _, pri in comps]
    return ll, sum(w * p for w, p in zip(ws, ps)) / s


def _group_mixture(nz, kz, zc, cp, a=.5, b=.5):
    """Like _split_mixture for any even number of cells: first half = XOR side 0, second half = side 1."""
    if cp is None:
        cp = (a, b, a, b)
    a1, b1, a0, b0 = cp[:4]; h = len(nz) // 2
    def part(c, a_, b_):
        return betaln(kz[c] + a_, nz[c] - kz[c] + b_) - betaln(a_, b_) if nz[c] else 0.0
    lA = sum(part(c, a1, b1) for c in range(h)) + sum(part(c, a0, b0) for c in range(h, 2 * h))
    lB = sum(part(c, a0, b0) for c in range(h)) + sum(part(c, a1, b1) for c in range(h, 2 * h))
    m = max(lA, lB); wA = math.exp(lA - m); wB = math.exp(lB - m); s = wA + wB
    ll = m + math.log(s / 2)
    def rate(c, on):
        a_, b_ = (a1, b1) if on else (a0, b0)
        return (kz[c] + a_) / (nz[c] + a_ + b_)
    if zc is None:
        n = sum(nz); f = [(nz[c] + 1 / (2 * h)) / (n + 1) for c in range(2 * h)]
        pA = sum(f[c] * rate(c, c < h) for c in range(2 * h)); pB = sum(f[c] * rate(c, c >= h) for c in range(2 * h))
    else:
        pA = rate(zc, zc < h); pB = rate(zc, zc >= h)
    return ll, (wA * pA + wB * pB) / s


def _tie_rows(rows6, single):
    """Merge 6 split cells (side x category) into 4 (side x state): state 1 = category `single`, 0 = the others."""
    def idx(c):
        return (c // 3) * 2 + int(c % 3 == single)
    out = []
    for nz, kz, zc in rows6:
        n4 = [0.0] * 4; k4 = [0.0] * 4
        for c in range(6):
            n4[idx(c)] += nz[c]; k4[idx(c)] += kz[c]
        out.append((n4, k4, None if zc is None else idx(zc)))
    return out


def _split_counts(seq_m, w, cut, taus, st, wt):
    i, j, k = st; ts, cs, ys = [], [], []
    for t in range(max(st), len(w)):
        if w[t] < 0 or w[t - i] < 0 or w[t - j] < 0 or seq_m[t - k] < 0:
            continue
        ts.append(t); cs.append((w[t - i] ^ w[t - j]) * 3 + _cat3(seq_m[t - k])); ys.append(w[t])
    n = [[0] for _ in range(6)]; kk = [[0] for _ in range(6)]
    for t, c, y in zip(ts, cs, ys):
        v = wt(t)
        for d in range(6):
            n[d].append(n[d][-1] + v * (c == d)); kk[d].append(kk[d][-1] + v * (c == d and y == 1))
    N = [x[-1] for x in n]; K = [x[-1] for x in kk]
    xc = None
    if cut - k >= 0 and all(w[cut - l] >= 0 for l in (i, j)) and seq_m[cut - k] >= 0:
        xc = (w[cut - i] ^ w[cut - j]) * 3 + _cat3(seq_m[cut - k])
    rows = []
    for tau in list(taus) + [None]:
        if tau is None:
            rows.append((N[:], K[:], xc)); continue
        q = bisect_left(ts, tau)
        nz = [n[d][q] + (N[_split_flip(d)] - n[_split_flip(d)][q]) for d in range(6)]
        kz = [kk[d][q] + (K[_split_flip(d)] - kk[_split_flip(d)][q]) for d in range(6)]
        rows.append((nz, kz, None if xc is None else _split_flip(xc)))
    return rows


def machine_counts(seq_m, cut, taus, ST, half_life=None):
    """Sufficient statistics, independent of any cell prior. Per structure: list over taus (+ None = no flip) of
    (nz, kz, zc): ticks and wins per gate cell (after the flip, if any) and the cell the cutoff falls in.
    Last entry: 'no gate' totals (n0, k0). half_life (ticks) optionally down-weights old ticks (counts become
    effective counts)."""
    w = wins(seq_m); out = []
    wt = (lambda t: 1.0) if half_life is None else (lambda t: 2.0 ** (-(cut - 1 - t) / half_life))
    for form, st in ST:
        if form == "xs":
            out.append(_split_counts(seq_m, w, cut, taus, st, wt)); continue
        if form == "xst":  # tied size split: lags (i, j, k, single category that forms its own state)
            out.append(_tie_rows(_split_counts(seq_m, w, cut, taus, st[:3], wt), st[3])); continue
        ts, xs, ys = [], [], []
        for t in range(max(st), len(w)):
            if w[t] < 0:
                continue
            x = gate_value(form, st, w, t)
            if x is None:
                continue
            ts.append(t); xs.append(x); ys.append(w[t])
        n = [[0], [0]]; k = [[0], [0]]
        for t, x, y in zip(ts, xs, ys):
            v = wt(t)
            for c in (0, 1):
                n[c].append(n[c][-1] + v * (x == c)); k[c].append(k[c][-1] + v * (x == c and y == 1))
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
            rows.append((nz, kz, None if xc is None else (xc if tau is None else xc ^ 1)))
        out.append(rows)
    obs = [(t, x) for t, x in enumerate(w) if x >= 0]
    out.append((sum(wt(t) for t, _ in obs), sum(wt(t) * x for t, x in obs)))
    return out


def tables_from_counts(C, ntau, a, b, cellprior=None):
    """(log marginal lik, P(win at cut)) per structure and tau, from machine_counts output."""
    out = []; base = 2 * betaln(a, b)
    for rows in C[:-1]:
        res = []
        for nz, kz, zc in rows:
            if len(nz) == 6:
                res.append(_split_mixture(nz, kz, zc, cellprior, a, b)); continue
            if len(nz) == 4:
                res.append(_group_mixture(nz, kz, zc, cellprior, a, b)); continue
            if cellprior is None:
                ll = betaln(kz[0] + a, nz[0] - kz[0] + b) + betaln(kz[1] + a, nz[1] - kz[1] + b) - base
                if zc is None:
                    pr = (kz[0] + kz[1] + a) / (nz[0] + nz[1] + a + b)
                else:
                    pr = (kz[zc] + a) / (nz[zc] + a + b)
            else:
                ll, pr = _cell_mixture(nz, kz, zc, cellprior)
            res.append((ll, pr))
        out.append(res)
    n0, k0 = C[-1]
    out.append([(betaln(k0 + a, n0 - k0 + b) - betaln(a, b), (k0 + a) / (n0 + a + b))] * ntau)
    return out


def machine_tables(seq_m, cut, taus, ST, a, b, cellprior=None, half_life=None):
    """Per structure (plus 'no gate' last): list over taus (+ None = no flip) of (log marginal lik, P(win at cut))."""
    return tables_from_counts(machine_counts(seq_m, cut, taus, ST, half_life), len(taus) + 1, a, b, cellprior)


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


def _combine(tb, sprior, lprior):
    J = []; PR = []
    for si, rows in enumerate(tb):
        if sprior[si] <= 0:
            continue
        ls = math.log(sprior[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lprior[ti]); PR.append(pr)
    z = logsumexp(J)
    return sum(math.exp(j - z) * p for j, p in zip(J, PR)), z


def _taus(cut, p_change, step, tau_min):
    taus = list(range(tau_min, cut - 2, step))
    return taus, [math.log(p_change / len(taus))] * len(taus) + [math.log(1 - p_change)]


def win_probability(q, seq_m, ST, sprior, a=.5, b=.5, p_change=.5, step=2, tau_min=40, cellprior=None,
                    evidence=False, half_life=None):
    """Posterior-mean P(win at cutoff). With evidence=True returns (P(win), log marginal likelihood of the log)."""
    taus, lprior = _taus(q["cutoff"], p_change, step, tau_min)
    tb = machine_tables(seq_m, q["cutoff"], taus, ST, a, b, cellprior, half_life)
    p, z = _combine(tb, sprior, lprior)
    return (p, z) if evidence else p


def win_probability_grid(q, seq_m, ST, sprior, cellpriors, a=.5, b=.5, p_change=.5, step=2, tau_min=40,
                         half_life=None):
    """[(P(win), log evidence)] for each cell prior in `cellpriors`, sharing one pass over the history."""
    taus, lprior = _taus(q["cutoff"], p_change, step, tau_min)
    C = machine_counts(seq_m, q["cutoff"], taus, ST, half_life)
    return [_combine(tables_from_counts(C, len(taus) + 1, a, b, cp), sprior, lprior) for cp in cellpriors]


def shift_cellprior(cp, delta, mode="both"):
    """Shift the on and/or off Beta means by delta on the log-odds scale, keeping strengths.
    mode: 'both' (same shift), 'on', 'off', 'contrast' (on up, off down)."""
    a1, b1, a0, b0 = cp; k1, k0 = a1 + b1, a0 + b0
    def sh(a_, k, d):
        m = a_ / k; m = min(max(m, 1e-6), 1 - 1e-6)
        m = 1 / (1 + math.exp(-(math.log(m / (1 - m)) + d)))
        return m * k, (1 - m) * k
    d_on = delta if mode in ("both", "on", "contrast") else 0.0
    d_off = {"both": delta, "off": delta, "contrast": -delta}.get(mode, 0.0)
    return sh(a1, k1, d_on) + sh(a0, k0, d_off)


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


SLOPES = (-.8, -.5, -.3, -.15, 0.0, .15, .3, .5, .8)


def _logit(m):
    m = min(max(m, 1e-6), 1 - 1e-6); return math.log(m / (1 - m))


def signal_cellprior(x, sig):
    """x = (mean_on, mean_off, k_on, k_off, slope_on, slope_off): means shifted on the log-odds scale by
    slope * signal (signal in {-1, 0, 1}); strengths shared."""
    mo, mf, ko, kf, so, sf = x
    a = 1 / (1 + math.exp(-(_logit(mo) + so * sig))); b = 1 / (1 + math.exp(-(_logit(mf) + sf * sig)))
    return (a * ko, (1 - a) * ko, b * kf, (1 - b) * kf)


def fit_signal_cell_prior(queries, seq, ST, sprior, cfg, step=4, rounds=2, cfg_fn=None):
    """Hierarchical cell prior per group with a log-odds slope in `signal` (round 5). Coordinate ascent of the
    summed train log-evidence over (on mean, off mean, on strength, off strength, on slope, off slope).
    cfg_fn(r) may return a per-machine cfg (e.g. short-log p_change)."""
    def kw(r):
        c = dict(cfg_fn(r) if cfg_fn else cfg); c["step"] = max(step, c.get("step", 2)); return c
    def ev(x):
        return sum(win_probability(r, seq[r["machine_id"]], ST, sprior, cellprior=signal_cellprior(x, r["signal"]),
                                   evidence=True, **kw(r))[1] for r in queries)
    x = [.85, .05, 12, 12, 0.0, 0.0]; cur = ev(x)
    grids = (MEANS_ON, MEANS_OFF, STRENGTHS, STRENGTHS, SLOPES, SLOPES)
    for _ in range(rounds):
        for i, grid in enumerate(grids):
            for v in grid:
                if v == x[i]:
                    continue
                y = x[:]; y[i] = v; e = ev(y)
                if e > cur:
                    x, cur = y, e
    return tuple(x), cur


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
