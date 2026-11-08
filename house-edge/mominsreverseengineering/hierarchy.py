"""Phase A: what is the generating distribution of machine win-rate parameters, and which parameters are shared?

TRAIN HISTORIES ONLY (no labels, no validation, no audit, no use of the model's own priors). Each machine's cell win rates are
INTEGRATED OUT on a logit grid, so counting noise is part of the likelihood and is not mistaken for generator spread.

Per machine: counts (wins k, ticks n) in 6 cells = gate parity (0/1) x payout category at the size lag (loss/1x/big), for each of
the 3 pairs the independent scan found ((1,3), (2,5), (4,7); size lag = smaller gate lag) plus a 'no gate' component.
Window: first 45% of a long log (polarity is constant there with high probability; flips sit near 68% of the cutoff),
all observed ticks of a short log. Latent: a_on, a_off (state-free base logits) ~ population prior; sub-cell logit =
a_role + gamma * s(c, d), d in {0,1} the unknown machine direction, role on/off unknown 50/50 (both averaged exactly).
Models are compared by maximized marginal log-likelihood (AIC / BIC / likelihood ratio):
  A  independent Beta (the v8 FORM, refit by maximum likelihood on this data, not copied)
  B  independent logit-normal           C  bivariate logit-normal (on/off correlation rho)
  D/E/F/G  C with sigma / signal slopes / gamma / all three SHARED across types
Run:  python3 hierarchy.py --data ../data/main [--models A,B,C,D,E,F,G]
"""
import argparse, math, os, time, zlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2
from common.data import public_dir, load_split

PAIRS = ((1, 3), (2, 5), (4, 7))
TYPES = ("balanced", "jackpot", "steady")
G = np.arange(-9.0, 7.0001, 0.2); NG = len(G)
SD = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])        # s(c, d): big category +1 for d = 0, flipped for d = 1
LOGSIG = -np.logaddexp(0, -G); LOG1MSIG = -np.logaddexp(0, G)


def extract(seq, end, t0=7):
    K = np.zeros((len(PAIRS), 2, 3)); N = np.zeros_like(K); kt = nt = 0
    w = [-1 if v < 0 else int(v > 0) for v in seq]
    for t in range(t0, end):
        if w[t] < 0:
            continue
        nt += 1; kt += w[t]
        for p, (i, j) in enumerate(PAIRS):
            if w[t - i] < 0 or w[t - j] < 0:
                continue
            x = w[t - i] ^ w[t - j]; c = 0 if seq[t - i] == 0 else (1 if seq[t - i] == 1 else 2)
            N[p, x, c] += 1; K[p, x, c] += w[t]
    return K, N, kt, nt


def load_machines(pub, frac=0.45, long_only=False):
    q, seq = load_split(pub, "train"); rows = []; Ks = []; Ns = []; kt = []; nt = []
    for r in q:
        s = seq[r["machine_id"]]; short = sum(v < 0 for v in s) > len(s) / 2
        if short and long_only:
            continue
        K, N, k, n = extract(s, len(s) if short else int(frac * len(s)))
        rows.append(dict(type=r["machine_type"], sig=r["signal"], ep=r["episode_id"], short=short, mid=r["machine_id"]))
        Ks.append(K); Ns.append(N); kt.append(k); nt.append(n)
    return rows, np.array(Ks), np.array(Ns), np.array(kt, float), np.array(nt, float)


def side_loglik(K, N, gamma):
    """[d] -> (M, P, 2 parities, NG): sum over the 3 categories of the binomial log-likelihood at logit g + gamma * s(c, d)."""
    out = []
    for d in (0, 1):
        th = G[None, None, None, None, :] + gamma * SD[d][None, None, None, :, None]
        lp = -np.logaddexp(0, -th); lq = -np.logaddexp(0, th)
        out.append((K[..., None] * lp + (N - K)[..., None] * lq).sum(axis=3))
    return out


class Hier:
    PER_TYPE = ("mu_on", "mu_off", "mu_n", "ls_n", "pi0", "pi1", "pi2")
    MAYBE_SHARED = ("ls_on", "ls_off", "b_on", "b_off", "gam", "rho")
    INIT = dict(mu_on=1.0, mu_off=-2.5, mu_n=0.0, ls_n=0.0, pi0=0.0, pi1=0.0, pi2=0.0, ls_on=math.log(.6), ls_off=math.log(.6),
                b_on=0.3, b_off=0.3, gam=0.4, rho=0.0)

    def __init__(self, data, prior="ln", rho=False, share=(), gam_fixed=None):
        self.rows, self.K, self.N, self.kt, self.nt = data
        self.gam_fixed = gam_fixed
        self.prior, self.use_rho, self.share = prior, rho, set(share)
        self.idx = {t: np.array([i for i, r in enumerate(self.rows) if r["type"] == t]) for t in TYPES}
        self.sig = np.array([r["sig"] for r in self.rows])
        self.names = []
        for n in self.PER_TYPE:
            self.names += [f"{n}|{t}" for t in TYPES]
        for n in self.MAYBE_SHARED:
            if n == "rho" and not rho:
                continue
            self.names += [n] if n in self.share else [f"{n}|{t}" for t in TYPES]
        if prior == "beta":
            self.INIT = dict(self.INIT, ls_on=math.log(25.0), ls_off=math.log(25.0))
        self.x0 = np.array([self.INIT[n.split("|")[0]] for n in self.names])
        if gam_fixed is not None:
            self.x0 = np.array([gam_fixed if n.split("|")[0] == "gam" else v for n, v in zip(self.names, self.x0)])
        self.bounds = []
        for n in self.names:
            b = n.split("|")[0]
            lo, hi = {"mu_on": (-8, 6), "mu_off": (-8, 6), "mu_n": (-8, 6), "ls_n": (-1, 1), "pi0": (-6, 6), "pi1": (-6, 6), "pi2": (-6, 6),
                      "b_on": (-2, 2), "b_off": (-2, 2), "gam": (0, 2), "rho": (-3, 3)}.get(b, (None, None))
            if b in ("ls_on", "ls_off"):
                lo, hi = (math.log(.15), math.log(2.5)) if prior == "ln" else (math.log(1.5), math.log(400.0))
            if b == "gam" and gam_fixed is not None:
                lo = hi = gam_fixed
            self.bounds.append((lo, hi))

    def get(self, x, name, t):
        d = dict(zip(self.names, x))
        return d[name] if name in d else d[f"{name}|{t}"]

    def W(self, x, t, sg):
        mo = self.get(x, "mu_on", t) + self.get(x, "b_on", t) * sg; mf = self.get(x, "mu_off", t) + self.get(x, "b_off", t) * sg
        lo, lf = self.get(x, "ls_on", t), self.get(x, "ls_off", t)
        if self.prior == "ln":
            so, sf = math.exp(lo), math.exp(lf); z1 = ((G - mo) / so)[:, None]; z2 = ((G - mf) / sf)[None, :]
            r = math.tanh(self.get(x, "rho", t)) if self.use_rho else 0.0
            Q = (z1 ** 2 - 2 * r * z1 * z2 + z2 ** 2) / (2 * (1 - r * r)); W = np.exp(-(Q - Q.min()))
        else:
            def wb(m_logit, lk):
                m = 1 / (1 + math.exp(-m_logit)); k = math.exp(lk); lw = m * k * LOGSIG + (1 - m) * k * LOG1MSIG
                w = np.exp(lw - lw.max()); return w / w.sum()
            W = np.outer(wb(mo, lo), wb(mf, lf))
        return W / W.sum()

    def type_ll(self, x, t):
        ix = self.idx[t]; K, N = self.K[ix], self.N[ix]; sg = self.sig[ix]
        L = side_loglik(K, N, self.get(x, "gam", t)); c = np.maximum(L[0].max(-1), L[1].max(-1))
        E = [np.exp(L[d] - c[..., None]) for d in (0, 1)]; val = np.zeros(K.shape[:2])
        for s in (-1, 0, 1):
            sel = sg == s
            if not sel.any():
                continue
            W = self.W(x, t, s); v = 0.0
            for d in (0, 1):
                e0 = E[d][sel][:, :, 0, :]; e1 = E[d][sel][:, :, 1, :]
                v = v + ((e0 @ W) * e1).sum(-1) + ((e1 @ W) * e0).sum(-1)
            val[sel] = v / 4
        lls = np.log(val + 1e-300) + c.sum(-1)
        kt, nt = self.kt[ix], self.nt[ix]; Ln = kt[:, None] * LOGSIG + (nt - kt)[:, None] * LOG1MSIG; cn = Ln.max(-1)
        mu, sd = self.get(x, "mu_n", t), math.exp(self.get(x, "ls_n", t)); wn = np.exp(-0.5 * ((G - mu) / sd) ** 2); wn /= wn.sum()
        lln = np.log(np.exp(Ln - cn[:, None]) @ wn + 1e-300) + cn
        lp = np.array([self.get(x, f"pi{p}", t) for p in range(3)] + [0.0]); lp = lp - logsumexp(lp)
        comp = np.concatenate([lls + lp[:3], (lln + lp[3])[:, None]], axis=1)
        return logsumexp(comp, axis=1).sum()

    def total(self, x):
        return sum(self.type_ll(x, t) for t in TYPES)


    def predict(self, x, t, sig, K, N, kt, nt, targ):
        """Posterior-predictive P(win) for one pseudo-query of type t. K, N: (P, 2, 3) counts from the observed prefix; targ[p] = (parity,
        category) of the target tick under pair p (from its observed inputs). Returns (P(win), structure posterior [3 pairs + none])."""
        gam = self.get(x, "gam", t); L = side_loglik(K[None], N[None], gam); c = np.maximum(L[0].max(-1), L[1].max(-1))[0]
        E = [np.exp(L[d][0] - c[..., None]) for d in (0, 1)]; W = self.W(x, t, sig)
        lpi = np.array([self.get(x, f"pi{p}", t) for p in range(3)] + [0.0]); lpi = lpi - logsumexp(lpi)
        sigm = lambda z: 1 / (1 + np.exp(-z)); logev = []; rate = []
        for p in range(len(PAIRS)):
            parity, cat = targ[p]; lz = []; rr = []
            for d in (0, 1):
                e0, e1 = E[d][p, 0], E[d][p, 1]
                for assign in (0, 1):
                    eon, eoff = (e0, e1) if assign == 0 else (e1, e0)
                    Pon = eon * (W @ eoff); Poff = eoff * (W.T @ eon); Z = Pon.sum()
                    Pr = Pon if (parity == 0) == (assign == 0) else Poff
                    lz.append(math.log(Z + 1e-300)); rr.append(float((Pr / Z * sigm(G + gam * SD[d][cat])).sum()))
            a = np.array(lz); m = a.max(); w = np.exp(a - m)
            logev.append(m + math.log(w.sum() / 4) + c[p].sum()); rate.append(float((w * np.array(rr)).sum() / w.sum()))
        Ln = kt * LOGSIG + (nt - kt) * LOG1MSIG; cn = Ln.max()
        mu, sd = self.get(x, "mu_n", t), math.exp(self.get(x, "ls_n", t)); wn = np.exp(-0.5 * ((G - mu) / sd) ** 2); wn /= wn.sum()
        post = np.exp(Ln - cn) * wn; zn = post.sum(); logev.append(math.log(zn + 1e-300) + cn); rate.append(float((post / zn * sigm(G)).sum()))
        lw = lpi + np.array(logev); lw = lw - logsumexp(lw); w = np.exp(lw)
        return float((w * np.array(rate)).sum()), w

    def fit(self, x0=None, maxiter=80):
        res = minimize(lambda x: -self.total(x), self.x0 if x0 is None else x0, method="L-BFGS-B", bounds=self.bounds,
                       options=dict(maxiter=maxiter, eps=1e-4, ftol=1e-10))
        self.x, self.ll, self.res = res.x, -res.fun, res
        return self

    @property
    def k(self):
        return len(self.names)


SPECS = {"A": dict(prior="beta"), "B": dict(prior="ln"), "C": dict(prior="ln", rho=True),
         "D": dict(prior="ln", rho=True, share=("ls_on", "ls_off")), "E": dict(prior="ln", rho=True, share=("b_on", "b_off")),
         "F": dict(prior="ln", rho=True, share=("gam",)), "G": dict(prior="ln", rho=True, share=("ls_on", "ls_off", "b_on", "b_off", "gam"))}
NESTED = {"C": "B", "D": "C", "E": "C", "F": "C", "G": "C"}


def describe(h):
    rows = []
    for t in TYPES:
        g = lambda n: h.get(h.x, n, t)
        sd = (math.exp(g("ls_on")), math.exp(g("ls_off"))) if h.prior == "ln" else (math.exp(g("ls_on")), math.exp(g("ls_off")))
        rho = f" rho {math.tanh(g('rho')):+.2f}" if h.use_rho else ""
        lab = "sigma" if h.prior == "ln" else "k"
        rows.append(f"    {t:9s} mu_on {g('mu_on'):+.2f}  mu_off {g('mu_off'):+.2f}  {lab}_on {sd[0]:.2f}  {lab}_off {sd[1]:.2f}{rho}  "
                    f"slope_on {g('b_on'):+.2f}  slope_off {g('b_off'):+.2f}  gamma {g('gam'):.2f}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--models", default="A,B,C,D,E,F,G")
    ap.add_argument("--long-only", action="store_true"); a = ap.parse_args()
    pub = public_dir(a.data); data = load_machines(pub, long_only=a.long_only)
    n = len(data[0]); print(f"machines used: {n} ({sum(r['short'] for r in data[0])} short), window = first 45% of long logs; "
                           f"train histories only; grid {NG} points", flush=True)
    fits = {}
    for m in a.models.split(","):
        t0 = time.time(); init = fits[NESTED[m]].x if m in NESTED and NESTED[m] in fits and m not in ("B",) else None
        h = Hier(data, **SPECS[m])
        if init is not None:                          # warm start from the parent model's parameters (same names where shared)
            d = dict(zip(fits[NESTED[m]].names, fits[NESTED[m]].x))
            h.x0 = np.array([d.get(nm, d.get(nm.split("|")[0] + "|" + TYPES[0], x0)) for nm, x0 in zip(h.names, h.x0)])
        h.fit(); fits[m] = h
        print(f"[{m}] loglik {h.ll:9.2f}  params {h.k:2d}  AIC {-2 * h.ll + 2 * h.k:9.1f}  BIC {-2 * h.ll + h.k * math.log(n):9.1f}  "
              f"({time.time() - t0:.0f}s, converged={h.res.success})", flush=True)
        print(describe(h), flush=True)
    print("\nLikelihood-ratio tests (nested):")
    for m, p in NESTED.items():
        if m in fits and p in fits:
            lr = 2 * (fits[m].ll - fits[p].ll) * -1 if False else 2 * (fits[p].ll - fits[m].ll)
            df = fits[p].k - fits[m].k if m != "C" else fits["C"].k - fits["B"].k
            if m == "C":
                lr = 2 * (fits["C"].ll - fits["B"].ll); print(f"  C vs B (adds rho, df {df}): 2*dLL = {lr:+.2f}, p = {chi2.sf(max(lr, 0), df):.4f}")
            else:
                print(f"  {m} (shared) vs {p} (free), df {df}: 2*dLL(free - shared) = {lr:+.2f}, p = {chi2.sf(max(lr, 0), df):.4f}  "
                      f"{'-> sharing REJECTED' if chi2.sf(max(lr, 0), df) < .05 else '-> sharing NOT rejected'}")
    if "A" in fits and "B" in fits:
        print(f"  Beta (A) vs logit-normal (B), same size: dLL = {fits['B'].ll - fits['A'].ll:+.2f} (positive favors logit-normal)")


if __name__ == "__main__":
    main()
