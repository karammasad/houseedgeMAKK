"""Structure test (2026-11-08): which parameter tying of the six size-split win-rate cells predicts held-out HISTORIES best?

Builds on Momin's shared-base structure (mominsreverseengineering/method4_ml/shared_base.py: extract(), derive_pairs(); imported read-only).
Per machine type, the population is fit by maximum marginal likelihood (machine rates integrated out on a logit grid, structure = one of the
derived lag pairs or 'no gate', on/off roles and state direction averaged), exactly as Momin's TypeFit, with ONE change per variant:
  A  six independent Beta cells (v8-style 'xs'): on-side Beta(a1, b1), off-side Beta(a0, b0) per cell
  B  Momin: two base rates (logit-normal per side) + one size-state shift gamma with profile (-1, -1, +1) over (loss, 1x, big)
  G  B with ONE gamma shared by all types (fit jointly: tested as B vs G through the summed held-out score with a common gamma)
  D  B + machine-specific gamma_m ~ N(gamma, tau^2) (Gauss-Hermite, 7 nodes; tau learned)
  E  B with separate gamma for the on and off side (rank-1 side x category interaction)
  P  B with a free category profile (g_loss, g_1x, g_big) instead of gamma * (-1, -1, +1)
Evaluation: 4-fold cross-fit BY EPISODE over train + validation + test HISTORIES (inputs only; no labels, targets or conditions): fit on
3 folds, score the held-out fold by log marginal likelihood (= sum of one-step-ahead log-losses, a proper out-of-sample predictive score).
Long logs use the first 45% (flip-free window, Momin's choice); short logs their whole history. Paired episode bootstrap on differences.
"""
import hashlib, math, os, sys, time
from multiprocessing import get_context
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from common.data import load_split
from mominsreverseengineering.method4_ml.shared_base import extract, derive_pairs, GRID, SD, LOGSIG, LOG1MSIG, SIG_LO, SIG_HI, lse

PUB = "data/main"; FOLDS = 4
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7); GH_LW = np.log(GH_W / GH_W.sum())
VARIANTS = ("A", "B", "D", "E", "P")


def lw(mu, s):
    z = -0.5 * ((GRID - mu) / s) ** 2; return z - lse(z, 0)


class Fit:
    """Population model per type. K, N: (M, P, 2, 3) counts; kt, nt: (M,) totals for 'no gate'; sig: (M,) signal."""
    def __init__(self, var, K, N, kt, nt, sig, P):
        self.var, self.K, self.N, self.kt, self.nt, self.sig, self.P = var, K, N, kt, nt, np.asarray(sig), P
        base = ["mu_n", "ls_n"] + [f"pi{p}" for p in range(P)]
        bb = [(-8, 6), (math.log(.2), math.log(2.0))] + [(-6, 6)] * P
        if var == "A":
            names = ["la1", "lb1", "la0", "lb0"]; bounds = [(-3, 6)] * 4
        else:
            names = ["mu_on", "mu_off", "ls_on", "ls_off", "b_on", "b_off"]; bounds = [(-8, 6), (-8, 6), (SIG_LO, SIG_HI), (SIG_LO, SIG_HI), (-1.5, 1.5), (-1.5, 1.5)]
            extra = {"B": (["gam"], [(0, 1.5)]), "D": (["gam", "ltau"], [(0, 1.5), (math.log(.01), math.log(1.0))]),
                     "E": (["gam_on", "gam_off"], [(0, 1.5), (-1.5, 1.5)]), "P": (["g0", "g1", "g2"], [(-1.5, 1.5)] * 3)}[var]
            names += extra[0]; bounds += extra[1]
        self.names, self.bounds = names + base, bounds + bb

    def shifts(self, v):
        """(2 sides, 3 categories) shift for direction +1 (direction -1 = negated)."""
        if self.var in ("B", "D"):
            return np.vstack([SD[0], SD[0]]) * v["gam"]
        if self.var == "E":
            return np.vstack([SD[0] * v["gam_on"], SD[0] * v["gam_off"]])   # row 0 = on side, row 1 = off side
        g = np.array([v["g0"], v["g1"], v["g2"]]); return np.vstack([g, g])

    def ev_logit(self, v, sh_on, sh_off):
        """Evidence (M, P) for given on/off-side shift vectors (3,), averaging on/off role assignment, integrating base rates."""
        M = self.K.shape[0]; ev = np.zeros((M, self.P)); so, sf = math.exp(v["ls_on"]), math.exp(v["ls_off"])
        def side(x, sh):     # (M, P, G): binomial log-lik of side x's three cells at each base logit, cells shifted by sh
            th = GRID[None, :] + sh[:, None]; lp = -np.logaddexp(0, -th); lq = -np.logaddexp(0, th)
            return np.einsum("mpc,cg->mpg", self.K[:, :, x, :], lp) + np.einsum("mpc,cg->mpg", (self.N - self.K)[:, :, x, :], lq)
        S = {(x, k): side(x, sh) for x in (0, 1) for k, sh in (("on", sh_on), ("off", sh_off))}
        for sg in (-1, 0, 1):
            sel = self.sig == sg
            if not sel.any():
                continue
            won, woff = lw(v["mu_on"] + v["b_on"] * sg, so), lw(v["mu_off"] + v["b_off"] * sg, sf)
            a = lse(S[(0, "on")][sel] + won, -1) + lse(S[(1, "off")][sel] + woff, -1)     # side 0 is 'on'
            b = lse(S[(0, "off")][sel] + woff, -1) + lse(S[(1, "on")][sel] + won, -1)     # side 1 is 'on'
            ev[sel] = np.logaddexp(a, b) - math.log(2)
        return ev

    def machine_ll(self, x):
        v = dict(zip(self.names, x)); M = self.K.shape[0]
        lpi = np.array([v[f"pi{p}"] for p in range(self.P)] + [0.0]); lpi = lpi - lse(lpi, 0)
        Ln = self.kt[:, None] * LOGSIG + (self.nt - self.kt)[:, None] * LOG1MSIG
        z = -0.5 * ((GRID - v["mu_n"]) / math.exp(v["ls_n"])) ** 2; none = lse(Ln + (z - lse(z, 0)), 1) + lpi[self.P]
        def mix(ev):
            return lse(np.concatenate([ev + lpi[:self.P], none[:, None]], 1), 1)
        if self.var == "A":
            a1, b1, a0, b0 = (math.exp(v[k]) for k in ("la1", "lb1", "la0", "lb0"))
            K, N = self.K, self.N
            def bb(a, b_):
                return (betaln(K + a, N - K + b_) - betaln(a, b_)).sum(-1)    # (M, P, 2)
            on, off = bb(a1, b1), bb(a0, b0)
            ev = np.logaddexp(on[:, :, 0] + off[:, :, 1], off[:, :, 0] + on[:, :, 1]) - math.log(2)
            return mix(ev)
        if self.var == "D":
            tau = math.exp(v["ltau"]); outs = []
            for xq, lwq in zip(GH_X, GH_LW):
                g = v["gam"] + tau * xq; sh = SD[0] * g
                outs.append(lwq + mix(np.logaddexp(self.ev_logit(v, sh, sh), self.ev_logit(v, -sh, -sh)) - math.log(2)))
            return lse(np.stack(outs, 0), 0)
        sh = self.shifts(v)
        return mix(np.logaddexp(self.ev_logit(v, sh[0], sh[1]), self.ev_logit(v, -sh[0], -sh[1])) - math.log(2))

    def fit(self):
        if self.var == "A":
            starts = [[math.log(2), math.log(1), math.log(.5), math.log(8)], [math.log(5), math.log(2), math.log(.3), math.log(15)]]
        else:
            core = {"B": [.4], "D": [.4, math.log(.2)], "E": [.4, .4], "P": [-.4, -.4, .4]}[self.var]
            starts = [[1.0, -2.5, math.log(.5), math.log(.5), .3, .3] + core, [2.0, -3.5, math.log(.35), math.log(.8), 0, 0] + [c * .5 for c in core]]
        best = None
        for s0 in starts:
            x0 = np.array(s0 + [0.0, 0.0] + [0.0] * self.P, float)
            r = minimize(lambda x: -self.machine_ll(x).sum(), x0, method="L-BFGS-B", bounds=self.bounds, options=dict(maxiter=400, eps=1e-4))
            if best is None or r.fun < best.fun:
                best = r
        return best.x, bool(best.success)


_D = {}


def _job(args):
    var, ty, fold = args; d = _D[ty]; tr = d["fold"] != fold; te = d["fold"] == fold
    def sub(m):
        return Fit(var, d["K"][m], d["N"][m], d["kt"][m], d["nt"][m], d["sig"][m], _D["P"])
    t0 = time.time(); x, ok = sub(tr).fit(); ll = sub(te).machine_ll(x)
    return var, ty, fold, x, ok, ll, time.time() - t0


def load_data(frac=0.45):
    from method4_ml.model import Model
    gm = Model()._fit_gate(PUB); pairs = derive_pairs(gm); data = {}
    for sp in ("train", "validation", "test"):
        q, seq = load_split(PUB, sp)
        for r in q:
            s = seq[r["machine_id"]]; short = sum(v < 0 for v in s) > len(s) / 2
            K, N, kt, nt = extract(s, len(s) if short else int(frac * len(s)), pairs)
            d = data.setdefault(r["machine_type"], {k: [] for k in ("K", "N", "kt", "nt", "sig", "ep", "short")})
            for k, val in (("K", K), ("N", N), ("kt", kt), ("nt", nt), ("sig", r["signal"]), ("ep", r["episode_id"]), ("short", short)):
                d[k].append(val)
    for d in data.values():
        for k in d:
            d[k] = np.array(d[k], dtype=float if k in ("K", "N", "kt", "nt") else None)
        d["fold"] = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % FOLDS for e in d["ep"]])
    return pairs, data


def main():
    pairs, data = load_data(); _D.update(data); _D["P"] = len(pairs); types = sorted(data)
    print("pairs derived from the fitted gate:", pairs, "| machines per type:", {t: len(data[t]["sig"]) for t in types}, flush=True)
    jobs = [(v, t, f) for v in VARIANTS for t in types for f in range(FOLDS)]
    with get_context("fork").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    ho = {}; conv = {}; params = {}
    for var, ty, fold, x, ok, ll, sec in res:
        m = data[ty]["fold"] == fold; ho.setdefault((var, ty), np.zeros(len(data[ty]["sig"])))[m] = ll
        conv.setdefault(var, []).append(ok); params.setdefault((var, ty), []).append(x)
    rng = np.random.default_rng(0)
    print("\nheld-out log marginal likelihood, difference vs B (Momin), nats; + = better than B; 95% CI by episode bootstrap")
    for var in VARIANTS:
        if var == "B":
            continue
        tot = []; line = []
        for ty in types:
            d = ho[(var, ty)] - ho[("B", ty)]; tot.append(d)
            line.append(f"{ty} {d.sum():+.1f}")
        D = np.concatenate(tot); EP = np.concatenate([data[t]["ep"] for t in types]); SH = np.concatenate([data[t]["short"] for t in types])
        ue, inv = np.unique(EP, return_inverse=True); epd = np.bincount(inv, D)
        bs = [epd[rng.integers(0, len(epd), len(epd))].sum() for _ in range(2000)]
        print(f"  {var}: total {D.sum():+.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]  short {D[SH.astype(bool)].sum():+.1f}  "
              f"long {D[~SH.astype(bool)].sum():+.1f}  | " + "  ".join(line) + f"  | converged {sum(conv[var])}/{len(conv[var])}")
    print(f"  (B converged {sum(conv['B'])}/{len(conv['B'])})")
    print("\nfitted shift parameters (mean over folds):")
    for ty in types:
        for var in ("B", "D", "E", "P"):
            f = Fit(var, *[data[ty][k][:1] for k in ("K", "N", "kt", "nt", "sig")], len(pairs)); x = np.mean(params[(var, ty)], 0)
            v = dict(zip(f.names, x)); keys = [k for k in ("gam", "ltau", "gam_on", "gam_off", "g0", "g1", "g2") if k in v]
            print(f"  {ty:9s} {var}: " + ", ".join(f"{k}={math.exp(v[k]) if k == 'ltau' else v[k]:.3f}" + (" (tau)" if k == "ltau" else "") for k in keys))
    # G: one gamma for all types -> refit B per type with gamma fixed at the pooled value is equivalent to testing whether per-type gammas differ:
    g = {ty: np.mean([dict(zip(Fit("B", *[data[ty][k][:1] for k in ("K", "N", "kt", "nt", "sig")], len(pairs)).names, x))["gam"] for x in params[("B", ty)]]) for ty in types}
    sd = {ty: np.std([dict(zip(Fit("B", *[data[ty][k][:1] for k in ("K", "N", "kt", "nt", "sig")], len(pairs)).names, x))["gam"] for x in params[("B", ty)]]) for ty in types}
    print("\nG (one gamma for all types?): per-type gamma across folds " + ", ".join(f"{t} {g[t]:.3f} +- {sd[t]:.3f}" for t in types))


if __name__ == "__main__":
    main()
