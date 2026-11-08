"""EGP: exact generator posterior for House Edge (clean-slate challenger; see DESIGN.md).

One joint probabilistic model of a machine's payout stream; prediction = its exact posterior predictive at the cutoff.
Discrete latents are summed exactly; per-machine base win logits are integrated on a grid; hyperparameters are fitted by
maximum marginal likelihood on HISTORIES only (no labels, targets or conditions are ever read here).

Latents per machine (type T, signal s):
  k   = (gate lag set G, size lag l)            prior pi[k] (global), support learned by screening
  d,e = size direction, win-shift direction     prior Pde[d, e] (global)
  p   = polarity (which raw gate cell is 'on' before the flip), 1/2 each
  tau = flip time (first tick with swapped polarity) or no flip; prior: rho[shape] no flip, else a position histogram of tau / cutoff
  mu_on ~ N(a_T + beta_T s, s_on_T), mu_off ~ N(c_T + beta'_T s, s_off_T); no-gate machines: mu_n ~ N(m_T, s_n_T)
Emission at tick t (b_t = 1[payout at t - l >= 2], g_t = XOR of wins at lags G, side_t = g_t xor p xor 1[t >= tau]):
  P(win) = sigmoid(mu_side + gamma_T (2 (b_t xor e) - 1));  size | win ~ V_T[b_t xor d]  (4-vector over 1x, 2x, 5x, 20x)
Ticks enter the likelihood only when the tick and all lags 1..MAXLAG before it are observed (one common tick set for every structure,
so structure evidences are comparable). Missing ticks are never treated as losses.
"""
import itertools as it, math
import numpy as np
from scipy.optimize import minimize

MAXLAG = 8
GRID = np.arange(-9.0, 7.0001, 0.2); NG = len(GRID)
SIZES = (1, 2, 5, 20)
TYPES_DEFAULT = ("balanced", "jackpot", "steady")
NBINS = 20
GATE_ORDER = 3       # largest XOR lag set screened (1 = singles, 2 = + pairs, 3 = + triples); kept only if the data give it mass
SIZE_DEFS = {"big": lambda y: y >= 2, "win": lambda y: y >= 1, "one": lambda y: y == 1}
PAIR_KEEP = True     # pair-level keep rule on top of the per-(gate, size lag) 1% rule
SIZE_CHOICES = ("big", "win", "one")
SIZE_DEF = "big"     # what the binary size state b_t reads at the size lag; chosen per kit by size evidence in fit_hyper
# Gate inputs (round-2 widening). A gate G is a tuple of input codes, XORed:
#   1..8            own WIN at that lag (the Round 1 family)
#   BIG + l         own "payout >= 2" bit at lag l                           -> "win chance driven by an earlier payout size"
#   NB + 10 r + l   WIN of the episode-mate r positions ahead (machine ids sorted, cyclic, r = 1..7) at lag l -> cross-machine rules
# Neighbour slots r are separate structures with their own learned prior mass, so "always the adjacent machine" and "some partner"
# are both expressible; a slot nobody uses is pruned by the screen and costs nothing.
BIG, NB = 100, 1000
CROSS = True         # screen neighbour inputs (singles and own-lag XOR neighbour-lag pairs)
BIGGATE = True       # screen own size-state singles as gate inputs
WINDOW_FRAC = 0.45   # flip-free prefix of long logs used for the population fit (lowered automatically if flips come earlier)
TICKSET = "own"     # "common": a tick counts only if all lags 1..MAXLAG are observed (pre-registered EGP)
                     # "own": a tick counts for structure k when k's own lags are observed; ticks k cannot use get a plug-in
                     #        marginal term (machine win rate, size frequencies) so every structure covers the same observed ticks


def lse(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def logsig(x):
    return -np.logaddexp(0.0, -x)


# ----------------------------------------------------------------------------------------------------------- data views
class Machine:
    """Arrays for one machine, built from its observed prefix y[0:cutoff] only."""
    __slots__ = ("qid", "ep", "mid", "type", "signal", "cutoff", "y", "short", "use", "w", "big", "cat", "obs8", "nbw", "b2")

    def __init__(self, qid, ep, mid, typ, signal, y):
        self.qid, self.ep, self.mid, self.type, self.signal = qid, ep, mid, typ, int(signal)
        self.y = np.asarray(y, dtype=np.int16); self.cutoff = len(self.y)
        obs = self.y >= 0
        self.short = bool(obs.sum() < 0.5 * self.cutoff)          # observable log shape
        self.w = np.where(obs, (self.y > 0).astype(np.int16), -1)
        self.big = np.where(obs, SIZE_DEFS[SIZE_DEF](self.y).astype(np.int16), -1)
        cat = np.full(self.cutoff, -1, np.int16)
        for i, v in enumerate(SIZES):
            cat[self.y == v] = i
        self.cat = cat
        # common usable tick set: tick observed and all lags 1..MAXLAG observed
        use = obs.copy(); use[:MAXLAG] = False
        for l in range(1, MAXLAG + 1):
            use[l:] &= obs[:-l]
        self.use = use
        self.obs8 = obs.copy(); self.obs8[:MAXLAG] = False
        self.b2 = np.where(obs, (self.y >= 2).astype(np.int16), -1)   # payout >= 2: the gate-input "size" bit (independent of SIZE_DEF)
        self.nbw = []                                            # episode-mates' win arrays by offset r = 1.. (set by link_episodes)


def link_episodes(machines):
    """Give each machine its episode-mates' WIN arrays (pre-cutoff observations only), ordered by cyclic offset in sorted machine id."""
    by = {}
    for m in machines:
        by.setdefault(m.ep, []).append(m)
    for ms in by.values():
        ms = sorted(ms, key=lambda m: m.mid); M = len(ms)
        for i, m in enumerate(ms):
            nb = []
            for r in range(1, M):
                w = ms[(i + r) % M].w; a = np.full(m.cutoff, -1, np.int16); n = min(len(w), m.cutoff); a[:n] = w[:n]; nb.append(a)
            m.nbw = nb
    return machines


def machines_from_split(queries, seq):
    return link_episodes([Machine(q["query_id"], q["episode_id"], q["machine_id"], q["machine_type"], q["signal"], seq[q["machine_id"]])
                          for q in queries])


def inp(m, q):
    """(array, lag) of gate input code q for machine m; -1 = unobserved."""
    if q < BIG:
        return m.w, q
    if q < NB:
        return m.b2, q - BIG
    r, l = divmod(q - NB, 10)
    return (m.nbw[r - 1] if r - 1 < len(m.nbw) else np.full(m.cutoff, -1, np.int16)), l


def is_own_win(G):
    return all(q < BIG for q in G)


def usable(m, G, l):
    if TICKSET == "common" or l is None:
        if is_own_win(G):
            return m.use
        u = m.use.copy()                                         # screen / common set + the extra inputs' availability
        for q in G:
            if q >= BIG:
                a, lg = inp(m, q); u[lg:] &= a[:-lg] >= 0
        return u
    u = m.obs8.copy()
    for q in set(G) | {l}:
        if q >= BIG:
            a, lg = inp(m, q); u[lg:] &= a[:-lg] >= 0
        else:
            u[q:] &= m.y[:-q] >= 0
    return u


def plugin_extra(m, G, l, end=None):
    """Log-likelihood of the observed ticks that structure (G, l) cannot use, under a plug-in marginal model
    (machine's own win rate and size frequencies over ticks < end). 0 for TICKSET 'common' and for complete logs."""
    if TICKSET == "common":
        return 0.0
    end = m.cutoff if end is None else end
    rest = m.obs8[:end] & ~usable(m, G, l)[:end]
    if not rest.any():
        return 0.0
    o = m.w[:end][m.w[:end] >= 0]; pbar = (o.sum() + 1) / (len(o) + 2)
    cw = m.cat[:end][m.w[:end] == 1]; vbar = (np.bincount(cw, minlength=4) + 1.0) / (len(cw) + 4)
    wr = m.w[:end][rest]; cr = m.cat[:end][rest]
    return float(wr.sum() * math.log(pbar) + (len(wr) - wr.sum()) * math.log(1 - pbar) + np.log(vbar[cr[wr == 1]]).sum())


def gate_bits(m, G, end=None, l=None):
    """Gate bit g_t (XOR of wins at lags G) for usable ticks t < end; G == () -> zeros."""
    end = m.cutoff if end is None else end
    t = np.nonzero(usable(m, G, l)[:end])[0]
    g = np.zeros(len(t), np.int16)
    for q in G:
        a, lg = inp(m, q); g ^= a[t - lg]
    return t, g


# ----------------------------------------------------------------------------------------------------------- screening
def all_gates():
    L = range(1, MAXLAG + 1)
    out = [()] + [G for k in range(1, GATE_ORDER + 1) for G in it.combinations(L, k)]
    if BIGGATE:
        out += [(BIG + l,) for l in L]
    if CROSS and TICKSET == "own":                             # needs the plug-in term for ticks a neighbour input cannot cover
        nb = [NB + 10 * r + l for r in range(1, 8) for l in L]
        out += [(q,) for q in nb] + [(a, q) for a in L for q in nb]
    return out


def gate_name(G):
    def one(q):
        if q < BIG:
            return str(q)
        if q < NB:
            return f"B{q - BIG}"
        r, l = divmod(q - NB, 10); return f"N{r}@{l}"
    return "(" + ",".join(one(q) for q in G) + ")"


def set_size_def(machines, d):
    """Rebuild every machine's size-state input for definition d (observed ticks only; missing stays -1)."""
    global SIZE_DEF
    SIZE_DEF = d
    for m in machines:
        m.big = np.where(m.y >= 0, SIZE_DEFS[d](m.y).astype(np.int16), -1)


def size_evidence(machines, d):
    """Total size log evidence under definition d: per machine, Dirichlet(1) per b group, averaged over size lags 1..MAXLAG."""
    tot = 0.0
    for m in machines:
        b_all = np.where(m.y >= 0, SIZE_DEFS[d](m.y).astype(np.int16), -1)
        t = np.nonzero(m.use[:window_end(m)])[0]; t = t[m.w[t] == 1]; ls = []
        for l in range(1, MAXLAG + 1):
            b = b_all[t - l]; c = m.cat[t]; ll = 0.0
            for bb in (0, 1):
                cnt = np.bincount(c[b == bb], minlength=4).astype(float)
                ll += sum(math.lgamma(x + 1) for x in cnt) - math.lgamma(cnt.sum() + 4) + math.lgamma(4)
            ls.append(ll)
        tot += float(lse(np.array(ls))) - math.log(MAXLAG)
    return tot


def _betaln(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def window_end(m, frac=None):
    return m.cutoff if m.short else int((WINDOW_FRAC if frac is None else frac) * m.cutoff)


def flip_quantile(H, q=0.05):
    """Quantile of the learned flip-position histogram (tau / cutoff), linear within bins."""
    c = np.cumsum(H.hist); b = int(np.searchsorted(c, q)); lo = c[b - 1] if b else 0.0
    return (b + (q - lo) / max(H.hist[b], 1e-12)) / NBINS


def screen(machines, min_mass=0.01, iters=60):
    """Learn the support of k = (gate lag set, size lag) on flip-free windows. Win evidence: Beta(1,1) per gate cell;
    size evidence: Dirichlet(1,1,1,1) per b group; joint prior over (G, l) by EM on the product evidence."""
    ALL_GATES = all_gates(); nG, nL = len(ALL_GATES), MAXLAG
    EW = np.zeros((len(machines), nG)); ES = np.zeros((len(machines), nL))
    for i, m in enumerate(machines):
        end = window_end(m)
        t0 = np.nonzero(m.use[:end])[0]; w0 = m.w[t0]; rows = {}
        for j, G in enumerate(ALL_GATES):
            g = np.zeros(len(t0), np.int16); ok = np.ones(len(t0), bool)
            for q in G:
                if q not in rows:
                    a, lg = inp(m, q); rows[q] = a[t0 - lg]
                v = rows[q]; g ^= v
                if q >= BIG:
                    ok &= v >= 0
            ll = 0.0
            for c in (0, 1):
                sel = ok & (g == c); n = int(sel.sum()); k = int(w0[sel].sum())
                ll += _betaln(k + 1, n - k + 1)
            if not ok.all():                                     # ticks this gate cannot read: one shared cell (same tick set for every gate)
                n = int((~ok).sum()); k = int(w0[~ok].sum()); ll += _betaln(k + 1, n - k + 1)
            EW[i, j] = ll
        t = np.nonzero(m.use[:end])[0]; t = t[m.w[t] == 1]
        for l in range(1, nL + 1):
            b = m.big[t - l]; c = m.cat[t]; ll = 0.0
            for bb in (0, 1):
                cnt = np.bincount(c[b == bb], minlength=4).astype(float)
                ll += sum(math.lgamma(x + 1) for x in cnt) - math.lgamma(cnt.sum() + 4) + math.lgamma(4)
            ES[i, l - 1] = ll
    E = EW[:, :, None] + ES[:, None, :]                          # (M, G, L)
    lp = np.full((nG, nL), -math.log(nG * nL))
    for _ in range(iters):
        r = E + lp[None]; r = np.exp(r - lse(r.reshape(len(machines), -1), 1)[:, None, None])
        lp = np.log(r.sum(0) + 1e-3) - math.log(r.sum() + 1e-3 * nG * nL)
    P = np.exp(lp)
    keep = [(ALL_GATES[a], b + 1, float(P[a, b])) for a in range(nG) for b in range(nL) if P[a, b] >= min_mass]
    # pair-level rule (Momin's round-2 safeguard): a gate whose TOTAL mass over size lags passes min_mass keeps its best size lag
    have = {(G, l) for G, l, _ in keep}
    for a in range(nG if PAIR_KEEP else 0):
        if P[a].sum() >= min_mass:
            b = int(np.argmax(P[a]))
            if (ALL_GATES[a], b + 1) not in have:
                keep.append((ALL_GATES[a], b + 1, float(P[a, b])))
    keep.sort(key=lambda x: -x[2])
    return keep, P


# ----------------------------------------------------------------------------------------------------------- sufficient statistics
def window_stats(m, K, end):
    """Win counts R[k, g, b, (n, w)] and size counts C[k, b, cat] over usable ticks t < end."""
    R = np.zeros((len(K), 2, 2, 2)); C = np.zeros((len(K), 2, 4)); E = np.zeros(len(K))
    for ki, (G, l) in enumerate(K):
        E[ki] = plugin_extra(m, G, l, end)
        t, g = gate_bits(m, G, end, l); b = m.big[t - l]; w = m.w[t]
        for gg in (0, 1):
            for bb in (0, 1):
                sel = (g == gg) & (b == bb)
                R[ki, gg, bb, 0] = sel.sum(); R[ki, gg, bb, 1] = w[sel].sum()
        tw = t[w == 1]; bw = m.big[tw - l]; cw = m.cat[tw]
        for bb in (0, 1):
            C[ki, bb] = np.bincount(cw[bw == bb], minlength=4)
    return R, C, E


# ----------------------------------------------------------------------------------------------------------- per-type hyperparameters
PNAMES = ["a_on", "a_off", "ls_on", "ls_off", "b_on", "b_off", "gam", "mu_n", "ls_n"] + [f"v{s}{c}" for s in (0, 1) for c in range(3)]
BOUNDS = [(-8, 6), (-8, 6), (math.log(.05), math.log(2.5)), (math.log(.05), math.log(2.5)), (-2, 2), (-2, 2), (0, 3),
          (-8, 6), (math.log(.05), math.log(2.5))] + [(-8, 8)] * 6


def unpack(x):
    v = dict(zip(PNAMES, x))
    V = np.zeros((2, 4))
    for s in (0, 1):
        z = np.array([0.0] + [v[f"v{s}{c}"] for c in range(3)]); V[s] = np.exp(z - lse(z))
    return v, V


def grid_logw(mu, sd):
    z = -0.5 * ((GRID - mu) / sd) ** 2
    return z - lse(z)


def basis(gam):
    """(4, NG): rows = coefficients for [k at h=1, (n-k) at h=1, k at h=0, (n-k) at h=0]."""
    th1 = GRID + gam; th0 = GRID - gam
    return np.stack([logsig(th1), logsig(-th1), logsig(th0), logsig(-th0)])


def counts4(nk_b, e):
    """nk_b (..., b, (n, k)) -> (..., 4) ordered as basis() with h = b xor e."""
    n1 = nk_b[..., 1 ^ e, 0]; k1 = nk_b[..., 1 ^ e, 1]; n0 = nk_b[..., e, 0]; k0 = nk_b[..., e, 1]
    return np.stack([k1, n1 - k1, k0, n0 - k0], -1)


class Hyper:
    """All fitted population parameters."""
    def __init__(self):
        self.K = []            # support [(G, l)]
        self.lpi = None        # log prior over K
        self.lPde = np.log(np.full((2, 2), .25))
        self.theta = {}        # type -> parameter vector
        self.rho = {False: .5, True: .1}   # P(no flip) by short flag
        self.hist = np.full(NBINS, 1.0 / NBINS)
        self.info = {}


def machine_terms(x, R, C, sig, K, lPde, E=None):
    """Per machine (window) log terms T[k, d, e, p] (no-gate structures: p axis duplicated) for one type's params x.
    R: (M, K, 2, 2, 2), C: (M, K, 2, 4), sig: (M,)."""
    v, V = unpack(x); B = basis(v["gam"]); lV = np.log(V)
    M = R.shape[0]; nK = len(K)
    S = np.zeros((M, nK, 2))                                     # size evidence per d
    for d in (0, 1):
        S[:, :, d] = (C[:, :, 0, :] * lV[0 ^ d]).sum(-1) + (C[:, :, 1, :] * lV[1 ^ d]).sum(-1)
    lw_on = np.stack([grid_logw(v["a_on"] + v["b_on"] * s, math.exp(v["ls_on"])) for s in (-1, 0, 1)])
    lw_off = np.stack([grid_logw(v["a_off"] + v["b_off"] * s, math.exp(v["ls_off"])) for s in (-1, 0, 1)])
    lw_n = grid_logw(v["mu_n"], math.exp(v["ls_n"]))
    si = sig + 1
    W = np.zeros((M, nK, 2, 2))                                  # [k, e, p]
    for ki, (G, l) in enumerate(K):
        for e in (0, 1):
            if len(G) == 0:
                cnt = counts4(R[:, ki, 0] + R[:, ki, 1], e)      # all ticks in one cell
                W[:, ki, e, :] = lse(cnt @ B + lw_n[None], 1)[:, None]
                continue
            L = counts4(R[:, ki], e) @ B                          # (M, g, NG)
            for p in (0, 1):
                W[:, ki, e, p] = lse(L[:, p] + lw_on[si], 1) + lse(L[:, 1 - p] + lw_off[si], 1)
    T = S[:, :, :, None, None] + W[:, :, None, :, :] + lPde[None, None, :, :, None] - math.log(2)
    if E is not None:
        T = T + E[:, :, None, None, None]
    return T


def fit_type(R, C, sig, K, lpi, lPde, x0, E=None):
    def nll(x):
        T = machine_terms(x, R, C, sig, K, lPde, E)
        return -float(lse(lse(T.reshape(T.shape[0], T.shape[1], -1), 2) + lpi[None], 1).sum())
    best = None
    for start in x0:
        r = minimize(nll, np.array(start, float), method="L-BFGS-B", bounds=BOUNDS, options=dict(maxiter=500, eps=1e-5, ftol=1e-11, gtol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    return best


def starts(machines):
    cats = np.concatenate([m.cat[m.w == 1] for m in machines]); f = np.bincount(cats[cats >= 0], minlength=4) + 1.0; f /= f.sum()
    z = np.log(f / f[0])[1:]
    lo = np.log(np.array([f[0] * 1.3, f[1], f[2], f[3]]) / (f[0] * 1.3))[1:]; hi = np.log(np.array([f[0] * .7, f[1] * 1.2, f[2] * 1.2, f[3] * 1.2]) / (f[0] * .7))[1:]
    s1 = [1.0, -2.5, math.log(.5), math.log(.5), .3, .3, .4, -1.0, 0.0] + list(lo) + list(hi)
    s2 = [0.0, -1.5, math.log(1.0), math.log(1.0), 0.0, 0.0, .1, 0.0, 0.0] + list(z) + list(z + .2)
    return [s1, s2]


def fit_hyper(machines, types=None, outer=3, em_iters=40, log=print):
    """Empirical-Bayes fit on the machines' histories (whatever set the caller passes: train only = inductive)."""
    H = Hyper()
    types = types or sorted({m.type for m in machines})
    ev = {d: size_evidence(machines, d) for d in SIZE_CHOICES}
    best = max(ev, key=ev.get); set_size_def(machines, best); H.size_def = best
    H.info["size_evidence"] = {d: round(v, 1) for d, v in ev.items()}
    log("  size state definition: " + ", ".join(f"{d} {v:.1f}" for d, v in ev.items()) + f" -> {best}")
    keep, P = screen(machines)
    H.K = [(G, l) for G, l, _ in keep]
    H.info["screen"] = [(gate_name(G), l, round(p, 4)) for G, l, p in keep]
    log(f"  screen support ({len(H.K)}, window {WINDOW_FRAC:.2f}): " + ", ".join(f"{gate_name(G)}/{l}:{p:.3f}" for G, l, p in keep))
    stats = [window_stats(m, H.K, window_end(m)) for m in machines]
    R = np.stack([s[0] for s in stats]); C = np.stack([s[1] for s in stats]); E = np.stack([s[2] for s in stats]); sig = np.array([m.signal for m in machines])
    ty = np.array([m.type for m in machines])
    H.lpi = np.log(np.array([p for _, _, p in keep]) / sum(p for _, _, p in keep))
    x0 = {t: starts([m for m in machines if m.type == t]) for t in types}
    # Alternate: continuous per-type parameters by L-BFGS (expensive) <-> discrete priors pi, Pde by EM with theta fixed (cheap).
    for it_ in range(outer):
        tot = 0.0
        for t in types:
            sel = ty == t
            r = fit_type(R[sel], C[sel], sig[sel], H.K, H.lpi, H.lPde, x0[t], E[sel])
            H.theta[t] = r.x; x0[t] = [r.x]; tot -= r.fun
            H.info.setdefault("converged", {})[t] = bool(r.success)
        for _ in range(em_iters):
            rk = np.zeros(len(H.K)); rde = np.zeros((2, 2)); ll = 0.0
            for t in types:
                sel = ty == t
                T = machine_terms(H.theta[t], R[sel], C[sel], sig[sel], H.K, H.lPde, E[sel]) + H.lpi[None, :, None, None, None]
                Z = lse(T.reshape(T.shape[0], -1), 1); ll += Z.sum()
                post = np.exp(T - Z[:, None, None, None, None])
                rk += post.sum((0, 2, 3, 4)); rde += post.sum((0, 1, 4))
            H.lpi = np.log((rk + 1e-2) / (rk.sum() + 1e-2 * len(rk)))
            H.lPde = np.log((rde + .5) / (rde.sum() + 2))
        H.info["loglik"] = float(ll)
        log(f"  outer {it_}: theta-step loglik {tot:.1f} -> after EM {ll:.1f}  pi " + " ".join(f"{math.exp(a):.3f}" for a in H.lpi) +
            "  Pde " + " ".join(f"{math.exp(a):.3f}" for a in H.lPde.ravel()))
    for t in types:
        v, V = unpack(H.theta[t])
        log(f"  {t:9s} " + " ".join(f"{k}={v[k]:+.3f}" for k in PNAMES[:9]) + "  V0 " + " ".join(f"{a:.3f}" for a in V[0]) + "  V1 " + " ".join(f"{a:.3f}" for a in V[1]))
    return H


# ----------------------------------------------------------------------------------------------------------- full-history inference
def machine_posterior(m, H, want_pred=True, shift=None, moments=None):
    """shift = (d_on, sd_on, d_off, sd_off): episode-effect prior override (mean offset added, prior sd replaced); None = default.
    moments: dict to receive the posterior mean / variance of mu_on, mu_off (gated structures only) relative to the type prior mean."""
    """Exact posterior over (k, d, e, p, tau) for one machine on its full observed prefix, and the 5-class predictive at m.cutoff.
    Returns (log evidence per tau slot [no-flip, tau = 0..cutoff], predictive 5-vector or None).
    Uses only m.y (ticks < cutoff)."""
    x = H.theta[m.type]; v, V = unpack(x); B = basis(v["gam"]); lV = np.log(V); gam = v["gam"]
    si = m.signal
    sh = shift or (0.0, math.exp(v["ls_on"]), 0.0, math.exp(v["ls_off"]))
    lw_on = grid_logw(v["a_on"] + v["b_on"] * si + sh[0], sh[1])
    lw_off = grid_logw(v["a_off"] + v["b_off"] * si + sh[2], sh[3])
    lw_n = grid_logw(v["mu_n"], math.exp(v["ls_n"]))
    tc = m.cutoff; nT = tc + 1                                    # tau slots 0..tc (tau = tc: only the predicted tick is flipped)
    sigp = 1 / (1 + np.exp(-(GRID + gam))); sigm = 1 / (1 + np.exp(-(GRID - gam)))
    logs = []; preds = []
    for ki, (G, l) in enumerate(H.K):
        t, g = gate_bits(m, G, None, l); b = m.big[t - l]; w = m.w[t]; Ek = plugin_extra(m, G, l)
        # size evidence
        tw = w == 1; cw = m.cat[t[tw]]; bw = b[tw]
        Cs = np.stack([np.bincount(cw[bw == bb], minlength=4) for bb in (0, 1)]).astype(float)
        S = np.array([(Cs[0] * lV[0 ^ d]).sum() + (Cs[1] * lV[1 ^ d]).sum() for d in (0, 1)])
        # cutoff-tick inputs (possibly missing -> average over the unknown bits)
        gs = [0] if len(G) == 0 else None
        if gs is None:
            vals = [a[tc - lg] for a, lg in (inp(m, q) for q in G)]
            gs = [int(np.bitwise_xor.reduce(np.array(vals)))] if min(vals) >= 0 else [0, 1]
        bs = [int(m.big[tc - l])] if m.big[tc - l] >= 0 else [0, 1]
        # cumulative counts by tau: cum[tau, g, b, (n, k)] = counts over usable ticks < tau
        oh = np.zeros((len(t), 2, 2, 2)); oh[np.arange(len(t)), g, b, 0] = 1; oh[np.arange(len(t)), g, b, 1] = w
        cum = np.zeros((nT, 2, 2, 2)); np.add.at(cum, t + 1, oh); cum = np.cumsum(cum, 0)
        tot = cum[-1]
        for e in (0, 1):
            if len(G) == 0:
                cnt = counts4((tot[0] + tot[1])[None], e)[0]
                Lg = cnt @ B + lw_n; le = lse(Lg); post = np.exp(Lg - le)
                lt = np.full(nT + 1, le)                          # no tau dependence
                pw = {bb: float(post @ (sigp if (bb ^ e) else sigm)) for bb in (0, 1)}
                for d in (0, 1):
                    base = S[d] + H.lPde[d, e] + H.lpi[ki] + Ek
                    for p in (0, 1):                               # keep the p axis for uniform bookkeeping
                        logs.append(base + lt - math.log(2))
                        preds.append([(bs, pw, d, None, None)])
                continue
            for p in (0, 1):
                # on-side counts: pre-tau ticks with g = p plus post-tau ticks with g = 1-p; off side the reverse
                on = cum[:, p] + (tot[1 - p] - cum[:, 1 - p])     # (nT, b, 2)
                off = cum[:, 1 - p] + (tot[p] - cum[:, p])
                Lon = counts4(on, e) @ B + lw_on; Loff = counts4(off, e) @ B + lw_off   # (nT, NG)
                lon = lse(Lon, 1); loff = lse(Loff, 1)
                lt_flip = lon + loff
                # no flip: tau beyond the cutoff = counts with everything pre-tau = slot nT-1 counts (tau = tc) but on-cell at cutoff = p
                lnf = lt_flip[-1]
                lt = np.concatenate([[lnf], lt_flip])
                # predictive P(win | config): which side is the cutoff tick on?
                post_on_nf = np.exp(Lon[-1] - lon[-1]); post_off_nf = np.exp(Loff[-1] - loff[-1])
                post_on = np.exp(Lon - lon[:, None]); post_off = np.exp(Loff - loff[:, None])
                for d in (0, 1):
                    base = S[d] + H.lPde[d, e] + H.lpi[ki] - math.log(2) + Ek
                    logs.append(base + lt)
                    preds.append([(bs, gs, d, e, p, post_on_nf, post_off_nf, post_on, post_off)])
    Lall = np.stack(logs)                                         # (configs, 1 + nT)
    # flip prior
    lpr = np.full(nT + 1, -np.inf)
    rho = H.rho[m.short]; lpr[0] = math.log(rho)
    u = np.arange(nT) / tc; bins = np.minimum((u * NBINS).astype(int), NBINS - 1)
    cnt = np.bincount(bins, minlength=NBINS)
    lpr[1:] = math.log(1 - rho) + np.log(H.hist[bins]) - np.log(cnt[bins])
    ev_tau = lse(Lall, 0)                                         # (1 + nT): evidence per tau slot (before the tau prior)
    if not want_pred:
        return ev_tau, None
    J = Lall + lpr[None]; Z = lse(J.ravel(), 0); Wt = np.exp(J - Z)
    out = np.zeros(5)
    for ci, item in enumerate(preds):
        item = item[0]
        if item[3] is None:                                       # no-gate structure
            bs, pw, d, _, _ = item
            wsum = Wt[ci].sum()
            for bb in bs:
                q = pw[bb]; wt = wsum / len(bs)
                out[0] += wt * (1 - q); out[1:] += wt * q * V[bb ^ d]
            continue
        bs, gs, d, e, p, pon_nf, poff_nf, pon, poff = item
        if moments is not None:
            for nm, pnf, pf in (("on", pon_nf, pon), ("off", poff_nf, poff)):
                for k, G_ in (("1", GRID), ("2", GRID ** 2)):
                    moments[nm + k] = moments.get(nm + k, 0.0) + Wt[ci, 0] * float(pnf @ G_) + float((Wt[ci, 1:] * (pf @ G_)).sum())
            moments["w"] = moments.get("w", 0.0) + float(Wt[ci].sum())
        for bb in bs:
            sg = sigp if (bb ^ e) else sigm
            for gg in gs:
                wt_nf = Wt[ci, 0] / (len(bs) * len(gs)); wt_f = Wt[ci, 1:] / (len(bs) * len(gs))
                # no flip: cutoff cell gg is on iff gg == p ; flipped (tau <= tc): on iff gg == 1 - p
                q_nf = float((pon_nf if gg == p else poff_nf) @ sg)
                q_f = (pon if gg == 1 - p else poff) @ sg          # (nT,)
                out[0] += wt_nf * (1 - q_nf) + float((wt_f * (1 - q_f)).sum())
                qw = wt_nf * q_nf + float((wt_f * q_f).sum())
                out[1:] += qw * V[bb ^ d]
    out = np.maximum(out, 0); out /= out.sum()
    if moments is not None and moments.get("w", 0) > 1e-9:
        W = moments["w"]
        for nm, mu0 in (("on", v["a_on"] + v["b_on"] * si), ("off", v["a_off"] + v["b_off"] * si)):
            e1 = moments[nm + "1"] / W; e2 = moments[nm + "2"] / W
            moments[nm] = (e1 - mu0 - (sh[0] if nm == "on" else sh[2]), max(e2 - e1 * e1, 1e-6))
    return ev_tau, out


def fit_flip_prior(machines, H, iters=50, log=print):
    """EM for rho[short], rho[long] and the tau / cutoff position histogram on FULL histories (other hyperparameters fixed)."""
    EV = [machine_posterior(m, H, want_pred=False)[0] for m in machines]
    for _ in range(iters):
        num = {False: [0.0, 0.0], True: [0.0, 0.0]}; hb = np.zeros(NBINS) + .1
        for m, ev in zip(machines, EV):
            tc = m.cutoff; nT = tc + 1
            u = np.arange(nT) / tc; bins = np.minimum((u * NBINS).astype(int), NBINS - 1); cnt = np.bincount(bins, minlength=NBINS)
            rho = H.rho[m.short]
            lpr = np.concatenate([[math.log(rho)], math.log(1 - rho) + np.log(H.hist[bins]) - np.log(cnt[bins])])
            r = ev + lpr; r = np.exp(r - lse(r, 0))
            num[m.short][0] += r[0]; num[m.short][1] += 1
            np.add.at(hb, bins, r[1:])
        for s in (False, True):
            if num[s][1]:
                H.rho[s] = float(np.clip((num[s][0] + .5) / (num[s][1] + 1), .01, .99))
        H.hist = hb / hb.sum()
    log(f"  flip prior: P(no flip) long {H.rho[False]:.3f} short {H.rho[True]:.3f}; position mode bin "
        f"{np.argmax(H.hist) / NBINS:.2f}-{(np.argmax(H.hist) + 1) / NBINS:.2f}, mass .55-.85 {H.hist[11:17].sum():.2f}")
    return H
