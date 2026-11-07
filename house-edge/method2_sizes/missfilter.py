"""Exact marginalization over unobserved gate inputs (Method 2, experiment 12). Standard library only.

Own implementation (does not use adi's or method 1's gate code). For one machine and one candidate gate (a, b):

  cell of tick t:      x_t = w_{t-a} XOR w_{t-b}       rate: r_on if x_t == o else r_off   (o in {0,1}: unknown orientation)
  KNOWN-input ticks    (both lags observed): contribute Binomial counts per cell, as every gate does.
  UNKNOWN-input ticks  (outcome observed, >= 1 lag unobserved): adi/method-1 style gates DISCARD these outcomes.
                       Here the hidden lag bits are integrated out: if no lag is observed, P(x_t = 1) = 2p(1-p);
                       if one lag has value v, P(x_t = 1) = p if v = 0 else 1 - p   (p = the machine's observed win rate, so
                       a hidden tick is never a loss and nonconsecutive ticks are never treated as adjacent).
                       Likelihood of the outcome: w ~ Bern(pi * r_on + (1 - pi) * r_off), pi = P(x_t == o).
  (r_on, r_off) are integrated on a G x G grid against the type's Beta cell priors (adi's learned signal-shifted
  priors are read as numbers only); (a, b, o) are averaged with a structure prior learned here by EM on TRAIN machines'
  pre-flip history (first 55%), so this expert does not inherit anyone else's structure weights.
  No flips (short logs flip ~5%). Output: posterior-mean P(win) at the cutoff, with and without the unknown-input ticks;
  the difference is the information the existing gates discard.
"""
import itertools, math

PAIRS = list(itertools.combinations(range(1, 9), 2))


def lse(v):
    m = max(v); return m + math.log(sum(math.exp(x - m) for x in v))


class MissFilter:
    def __init__(self, signal_priors, G=18):
        self.sp, self.G = signal_priors, G
        self.g = [(i + .5) / G for i in range(G)]
        self.lg = [math.log(x) for x in self.g]; self.l1g = [math.log(1 - x) for x in self.g]
        self.prior = None

    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    # ---- priors on the grid -------------------------------------------------------------------------------------
    def _grid_prior(self, typ, sig):
        mo, mf, ko, kf, so, sf = self.sp[typ]
        lg = lambda p: math.log(p / (1 - p))
        a = self.sig(lg(mo) + so * sig); b = self.sig(lg(mf) + sf * sig)
        def beta_w(m, k):
            al, be = m * k, (1 - m) * k
            w = [math.exp((al - 1) * math.log(x) + (be - 1) * math.log(1 - x)) for x in self.g]; z = sum(w)
            return [math.log(x / z) for x in w]
        return beta_w(a, ko), beta_w(b, kf)

    # ---- per-machine evidence ------------------------------------------------------------------------------------
    @staticmethod
    def _split(w, pair, p):
        """Counts for known-input ticks and grouped unknown-input ticks for gate `pair`."""
        a, b = pair; n = [0, 0]; k = [0, 0]; groups = {}
        for t in range(b, len(w)):
            if w[t] < 0:
                continue
            va, vb = w[t - a], w[t - b]
            if va >= 0 and vb >= 0:
                x = va ^ vb; n[x] += 1; k[x] += w[t]
            else:
                if va < 0 and vb < 0:
                    p1 = 2 * p * (1 - p)
                else:
                    v = va if va >= 0 else vb; p1 = p if v == 0 else 1 - p
                key = (round(p1, 4), w[t]); groups[key] = groups.get(key, 0) + 1
        return n, k, groups

    def _evidence(self, w, pair, o, pr_on, pr_off, use_unknown, p):
        """log-sum over the grid of prior * likelihood; returns (log marginal, posterior-mean r_on, r_off arrays)."""
        n, k, groups = self._split(w, pair, p); G = self.G
        # known part separates in (r_on, r_off)
        con = [k[o] * self.lg[i] + (n[o] - k[o]) * self.l1g[i] + pr_on[i] for i in range(G)]
        cof = [k[1 - o] * self.lg[j] + (n[1 - o] - k[1 - o]) * self.l1g[j] + pr_off[j] for j in range(G)]
        gl = list(groups.items()) if use_unknown else []
        post = []   # (log weight, i, j)
        for i in range(G):
            gi = self.g[i]
            for j in range(G):
                lw = con[i] + cof[j]; gj = self.g[j]
                for (p1, wt), cnt in gl:
                    pi = p1 if o == 1 else 1 - p1
                    q = pi * gi + (1 - pi) * gj
                    lw += cnt * math.log(q if wt else 1 - q)
                post.append((lw, i, j))
        return lse([x for x, _, _ in post]), post

    def machine(self, r, s, use_unknown=True, prior=None, want_pred=True):
        """Per-hypothesis log evidence and P(win) at the cutoff. Hypotheses: (pair, o) and 'none'."""
        w = [-1 if v < 0 else int(v > 0) for v in s]; obs = [x for x in w if x >= 0]
        p = min(.95, max(.05, sum(obs) / max(1, len(obs)))); pr_on, pr_off = self._grid_prior(r["machine_type"], r["signal"])
        cut = len(s); out = []
        for pair in PAIRS:
            a, b = pair
            va, vb = w[cut - a], w[cut - b]; xc = (va ^ vb) if va >= 0 and vb >= 0 else None
            for o in (0, 1):
                z, post = self._evidence(w, pair, o, pr_on, pr_off, use_unknown, p)
                pwin = None
                if want_pred:
                    m = max(x for x, _, _ in post); den = sum(math.exp(x - m) for x, _, _ in post); num = 0.0
                    for lw, i, j in post:
                        rate = self.g[i] if (xc == o) else self.g[j] if xc is not None else (.5 * (self.g[i] + self.g[j]))
                        num += math.exp(lw - m) * rate
                    pwin = num / den
                out.append((z + math.log(.5), pwin))
        # 'none': a constant rate with the type's mean cell prior
        n1 = sum(obs); n0 = len(obs) - n1; m_t = .5 * (self.g[0] * 0 + 1) * 0 + (self.none_mean.get(r["machine_type"], .4))
        al, be = 4 * m_t, 4 * (1 - m_t)
        lb = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
        out.append((lb(al + n1, be + n0) - lb(al, be), (al + n1) / (al + be + len(obs))))
        return out

    def fit(self, q, seq, frac=.55):
        """Structure prior over (pair or none) by EM on train machines' pre-flip history (first `frac`)."""
        tot = {}; cnt = {}
        for r in q:
            s = seq[r["machine_id"]]; ok = [v for v in s if v >= 0]
            if ok:
                tot[r["machine_type"]] = tot.get(r["machine_type"], 0) + sum(v > 0 for v in ok); cnt[r["machine_type"]] = cnt.get(r["machine_type"], 0) + len(ok)
        self.none_mean = {t: tot[t] / cnt[t] for t in tot}
        K = len(PAIRS) * 2 + 1; ev = []
        for r in q:
            s = seq[r["machine_id"]]
            if sum(v >= 0 for v in s) < 150:
                continue
            seg = [v if t < int(frac * len(s)) else -1 for t, v in enumerate(s)]
            ev.append([z for z, _ in self.machine(r, seg, use_unknown=False, want_pred=False)])
        pi = [1 / K] * K
        for _ in range(40):
            acc = [0.0] * K
            for row in ev:
                lp = [z + math.log(max(1e-300, p)) for z, p in zip(row, pi)]; zz = lse(lp)
                for j, x in enumerate(lp):
                    acc[j] += math.exp(x - zz)
            pi = [max(1e-4, x / len(ev)) for x in acc]; zz = sum(pi); pi = [x / zz for x in pi]
        self.prior = pi
        return self

    def pwin(self, r, s, use_unknown=True):
        res = self.machine(r, s, use_unknown); lp = [z + math.log(p) for (z, _), p in zip(res, self.prior)]
        zz = lse(lp)
        return sum(math.exp(x - zz) * pw for x, (_, pw) in zip(lp, res))

    def top(self, k=4):
        names = [f"{p}/o{o}" for p in PAIRS for o in (0, 1)] + ["none"]
        return sorted(zip(self.prior, names), reverse=True)[:k]
