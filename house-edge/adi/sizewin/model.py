"""adi round 7: size-state win correction on top of the team FINAL (standard library). Owner: Adi.

Inspired by method4_ml v5's finding that its stacker gains by seeing the size model's P(1x | win): the hidden size
state also moves the win rate (adi round 4). Instead of a 770-tree black box, this is a small logistic correction:

  logit P'(win) = b0 + b1 * logit P(win) + sum_type [c_on,t * d * P(win) + c_off,t * d * (1 - P(win))]

where P(win) is FINAL's win probability (adi gate) and d = the size model's P(1x | win) minus its type mean. Eight
numbers, fitted by L2-regularized logistic regression (Newton's method, pure Python) on train HISTORY ticks: for
sampled ticks t, features use only ticks < t and the label is whether tick t won (no validation labels/targets).
Short-history masked copies (40 recent + 24 older ticks, like the short_history condition) are added so the
correction also learns short-log behaviour. Sizes are FINAL's, unchanged.

  python -m common.evaluate --method adi.sizewin --data data/main
"""
import math, random
from multiprocessing import get_context
from common.data import load_split, wins
import method2_sizes.round2 as R
from method2_sizes.model import Model as M2

TYPES = ("steady", "balanced", "jackpot")
_G = {}


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))


def _feat(base, r, s):
    """[logit pw, (d*pw, d*(1-pw)) for the machine's type, zeros elsewhere], plus pw itself."""
    pw = base.win.win_probs([r], {r["machine_id"]: s})[r["query_id"]]
    a = base.sizes.predict(r, s)[0]; d = a - base.sizes.mu.get(r["machine_type"], a)
    f = [1.0, _logit(pw)] + [0.0] * (2 * len(TYPES))
    i = TYPES.index(r["machine_type"]); f[2 + 2 * i] = d * pw; f[3 + 2 * i] = d * (1 - pw)
    return f, pw


def _job(args):
    r, s, t = args
    rr = dict(r, cutoff=t, query_id=f'{r["query_id"]}@{t}')
    return _feat(_G["base"], rr, s[:t])[0]


class Model:
    def __init__(self, base=None, K=8, K_short=4, t_min=100, l2=1.0, splits=("train",), workers=None, seed=0,
                 free_base=False):
        self.base_kw = dict(base) if base is not None else dict(R.FINAL)
        self.K, self.K_short, self.t_min, self.l2, self.splits, self.seed = K, K_short, t_min, l2, splits, seed
        self.workers = workers
        self.free_base = free_base  # False: keep FINAL's logit as a fixed offset (b0=0, b1=1); learn size terms only

    def _examples(self, pub):
        jobs, ys = [], []
        for split in self.splits:
            q, seq = load_split(pub, split)
            for r in q:
                s = seq[r["machine_id"]]; w = wins(s); cut = r["cutoff"]
                rng = random.Random(f'{self.seed}|{r["machine_id"]}')
                if min(s) < 0:   # already-short logs: use only fully observed recent ticks
                    continue
                for t in sorted(rng.sample(range(self.t_min, cut), min(self.K, cut - self.t_min))):
                    if w[t] >= 0:
                        jobs.append((r, s, t)); ys.append(w[t])
                lo = max(self.t_min, 64)
                for t in sorted(rng.sample(range(lo, cut), min(self.K_short, cut - lo))):
                    keep = set(range(t - 40, t)) | set(rng.sample(range(t - 40), 24))
                    ms = [x if i in keep else -1 for i, x in enumerate(s[:t])] + s[t:]
                    jobs.append((r, ms, t)); ys.append(w[t])
        return jobs, ys

    def fit(self, pub):
        self.base = M2(**self.base_kw).fit(pub)
        jobs, y = self._examples(pub)
        _G["base"] = self.base
        import os
        n = self.workers or min(os.cpu_count() or 1, 8)
        with get_context("fork").Pool(n) as pool:
            X = pool.map(_job, jobs, chunksize=16)
        self.n_train = len(X)
        self.coef = self._logreg(X, y)
        return self

    def _logreg(self, X, y, iters=25):
        """L2-penalized logistic regression toward the identity correction (b0=0, b1=1, c=0); Newton steps."""
        k = len(X[0]); prior = [0.0, 1.0] + [0.0] * (k - 2); b = prior[:]
        for _ in range(iters):
            lam = [self.l2 if (self.free_base or j >= 2) else 1e9 for j in range(k)]
            g = [lam[j] * (b[j] - prior[j]) for j in range(k)]
            H = [[lam[i] if i == j else 0.0 for j in range(k)] for i in range(k)]
            for x, yy in zip(X, y):
                z = sum(bj * xj for bj, xj in zip(b, x)); p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
                for i in range(k):
                    g[i] += (p - yy) * x[i]
                    wi = p * (1 - p) * x[i]
                    if wi:
                        for j in range(k):
                            H[i][j] += wi * x[j]
            step = _solve(H, g)
            b = [bj - sj for bj, sj in zip(b, step)]
            if max(abs(sj) for sj in step) < 1e-9:
                break
        return b

    def predict(self, pub, split):
        base = self.base.predict(pub, split)
        q, seq = load_split(pub, split); out = {}
        for r in q:
            s = seq[r["machine_id"]]; f, _ = _feat(self.base, r, s)
            z = sum(bj * xj for bj, xj in zip(self.coef, f)); pw = 1 / (1 + math.exp(-z))
            p = base[r["query_id"]]; pw0 = 1 - p[0]
            out[r["query_id"]] = [1 - pw] + [pw * x / pw0 for x in p[1:]] if pw0 > 0 else p
        return out


def _solve(A, b):
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] if M[i][i] else 0.0 for i in range(n)]
