"""Two-level hierarchical cell model for size-split gates (adi "cook"). Standard library only.

adi's size-split gate ("xs") splits each XOR side into 3 sub-cells (payout category at the split lag: loss/1x/big).
Today each sub-cell's win rate is an independent draw from the type's on/off Beta prior, so a sub-cell with few
ticks shrinks to the TYPE mean even when the machine's other sub-cells on the same side say otherwise.

Here the prior is two-level:
    side rate      mu   ~ Beta(a, b)                       (the learned on/off type prior, unchanged)
    sub-cell rate  p_c  ~ Beta(mu * kappa, (1 - mu) * kappa) independently per sub-cell, given mu
so sub-cells borrow strength from the same machine first. mu is marginalized exactly on a logit grid; given mu each
sub-cell's marginal is closed form (beta-binomial). kappa -> inf is the tied (one rate per side) model; small kappa
is nearly free sub-cells. Everything else (structure x flip-time averaging, 50/50 on-side mixture) is unchanged.
"""
import math
import adi.gate as G
from method1_gate.gate import betaln

GRID_N = 40
_ORIG = G._split_mixture
STATE = {"kappa": None, "short": None, "long": None, "cur": None}
_GRID = {}   # (a, b, kappa) -> (mus, logw)
_L = {}      # (a, b, kappa, n, k) -> [log marginal of a sub-cell with n ticks, k wins, per grid point]


def _lse(v):
    m = max(v); return m + math.log(sum(math.exp(x - m) for x in v))


def _grid(a, b, kappa):
    key = (a, b, kappa)
    g = _GRID.get(key)
    if g is None:
        xs = [-9.0 + 18.0 * (i + .5) / GRID_N for i in range(GRID_N)]
        mus = [1 / (1 + math.exp(-x)) for x in xs]
        # Beta(a, b) density in logit space is mu^a (1 - mu)^b up to a constant
        lw = [a * math.log(m) + b * math.log(1 - m) for m in mus]
        z = _lse(lw); lw = [x - z for x in lw]
        g = _GRID[key] = (mus, lw)
    return g


def _vec(a, b, kappa, n, k):
    key = (a, b, kappa, n, k)
    v = _L.get(key)
    if v is None:
        mus, _ = _grid(a, b, kappa)
        v = _L[key] = [betaln(k + m * kappa, n - k + (1 - m) * kappa) - betaln(m * kappa, (1 - m) * kappa)
                       for m in mus]
        if len(_L) > 2_000_000:
            _L.clear()
    return v


def _side(nz, kz, cells, a, b, kappa):
    """(log marginal of one side's sub-cells, posterior weights over the mu grid)."""
    mus, lw = _grid(a, b, kappa)
    acc = list(lw)
    for c in cells:
        if nz[c]:
            v = _vec(a, b, kappa, nz[c], kz[c])
            acc = [x + y for x, y in zip(acc, v)]
    z = _lse(acc)
    return z, [math.exp(x - z) for x in acc]


class TCP(tuple):
    """Cell prior tuple tagged with the machine type, so the split mixture can use a per-type kappa."""
    mtype = None


def tag(cp, mtype):
    if cp is None:
        return None
    t = TCP(cp); t.mtype = mtype; return t


def hier_split_mixture(nz, kz, zc, cp, a=.5, b=.5):
    kappa = STATE["cur"]
    if isinstance(kappa, dict):
        kappa = kappa.get(getattr(cp, "mtype", None))
    if kappa is None:
        return _ORIG(nz, kz, zc, cp, a, b)
    if cp is None:
        cp = (a, b, a, b)
    a1, b1, a0, b0 = cp
    lo, hi = (0, 1, 2), (3, 4, 5)
    # assignment A: side 0 (cells 0-2) on, side 1 off; B: the reverse
    zA0, wA0 = _side(nz, kz, lo, a1, b1, kappa); zA1, wA1 = _side(nz, kz, hi, a0, b0, kappa)
    zB0, wB0 = _side(nz, kz, lo, a0, b0, kappa); zB1, wB1 = _side(nz, kz, hi, a1, b1, kappa)
    lA = zA0 + zA1; lB = zB0 + zB1
    m = max(lA, lB); WA = math.exp(lA - m); WB = math.exp(lB - m); s = WA + WB
    ll = m + math.log(s / 2)

    def rate(c, w, pa, pb):
        mus, _ = _grid(pa, pb, kappa)
        return sum(wi * (kz[c] + mu * kappa) / (nz[c] + kappa) for wi, mu in zip(w, mus))

    def pred(c):
        if c < 3:
            return rate(c, wA0, a1, b1), rate(c, wB0, a0, b0)
        return rate(c, wA1, a0, b0), rate(c, wB1, a1, b1)

    if zc is None:
        n = sum(nz); f = [(nz[c] + 1 / 6) / (n + 1) for c in range(6)]
        pA = pB = 0.0
        for c in range(6):
            x, y = pred(c); pA += f[c] * x; pB += f[c] * y
    else:
        pA, pB = pred(zc)
    return ll, (WA * pA + WB * pB) / s


_ORIG_MT = G.machine_tables
_ORIG_WPG = G.win_probability_grid


def _set(seq_m):
    STATE["cur"] = STATE["short"] if min(seq_m, default=0) < 0 else STATE["long"]


def _machine_tables(seq_m, *a, **k):
    _set(seq_m); return _ORIG_MT(seq_m, *a, **k)


def _win_probability_grid(q, seq_m, *a, **k):
    _set(seq_m); return _ORIG_WPG(q, seq_m, *a, **k)


def install(kappa_short, kappa_long=None):
    """Hierarchical split cells with kappa_short on short logs (any missing tick) and kappa_long on long logs;
    None keeps adi's independent sub-cells. Patches adi.gate in-process (fork-safe)."""
    f = lambda x: x if (x is None or isinstance(x, dict)) else float(x)
    STATE["short"], STATE["long"] = f(kappa_short), f(kappa_long)
    on = any(STATE[k] not in (None, {}) for k in ("short", "long"))
    G._split_mixture = hier_split_mixture if on else _ORIG
    G.machine_tables = _machine_tables if on else _ORIG_MT
    G.win_probability_grid = _win_probability_grid if on else _ORIG_WPG


_ORIG_SHIFT = G.shift_cellprior


def _shift(cp, delta, mode="both"):
    return tag(_ORIG_SHIFT(cp, delta, mode), getattr(cp, "mtype", None))


def tag_adi():
    """Make adi.model.Model._cp return type-tagged priors (and keep tags through shift_cellprior)."""
    import adi.model as AM
    if getattr(AM.Model._cp, "_cook", False):
        return
    orig = AM.Model._cp
    def _cp(self, r):
        return tag(orig(self, r), r["machine_type"])
    _cp._cook = True
    AM.Model._cp = _cp
    G.shift_cellprior = _shift
