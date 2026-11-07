"""Payout-size models given a win (standard library only). Owner: Person B.

A size model maps a machine's history to a distribution over (1x, 2x, 5x, 20x) given a win.
Interface:  SizeModel.fit(train_queries, train_seq) -> self ;  SizeModel.predict(query, seq_m) -> [s1, s2, s5, s20]
"""
import math
from collections import defaultdict
from common.data import SIZES
from method1_gate.gate import structures, machine_tables, learn_structure_prior, logsumexp as _logsumexp


def size_counts(seq_m):
    return [sum(1 for x in seq_m if x == v) for v in SIZES]


class DirichletSizes:
    """Shrink each machine's counts toward its type x signal average (strength = pseudo-counts)."""
    def __init__(self, strength=30.0):
        self.strength = strength

    def fit(self, queries, seq):
        d = defaultdict(lambda: [0, 0, 0, 0])
        for r in queries:
            c = d[(r["machine_type"], r["signal"])]
            for i, x in enumerate(size_counts(seq[r["machine_id"]])):
                c[i] += x
        self.prior = {k: [(x + 1) / (sum(v) + 4) for x in v] for k, v in d.items()}
        return self

    def predict(self, q, seq_m):
        pri = self.prior[(q["machine_type"], q["signal"])]; c = size_counts(seq_m); n = sum(c)
        return [(x + self.strength * p) / (n + self.strength) for x, p in zip(c, pri)]


def _digamma(x):
    r = 0.0
    while x < 6:
        r -= 1 / x; x += 1
    f = 1 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


class MinkaSizes:
    """B6 control: Dirichlet-multinomial per machine type, alpha fitted by Minka's fixed-point iteration
    on train machines' size counts (strength and mean both learned). Posterior mean = (c + alpha)/(n + sum alpha)."""
    def __init__(self, iters=500):
        self.iters = iters

    def fit(self, queries, seq):
        C = defaultdict(list)
        for r in queries:
            c = size_counts(seq[r["machine_id"]])
            if sum(c) > 0:
                C[r["machine_type"]].append(c)
        self.alpha = {}
        for t, cs in C.items():
            tot = [sum(c[i] for c in cs) for i in range(4)]; N = sum(tot)
            a = [10 * (x + 1) / (N + 4) for x in tot]
            for _ in range(self.iters):
                A = sum(a)
                den = sum(_digamma(sum(c) + A) - _digamma(A) for c in cs)
                a = [max(1e-6, a[i] * sum(_digamma(c[i] + a[i]) - _digamma(a[i]) for c in cs) / den) for i in range(4)]
            self.alpha[t] = a
        return self

    def predict(self, q, seq_m):
        a = self.alpha[q["machine_type"]]; c = size_counts(seq_m); s = sum(c) + sum(a)
        return [(x + y) / s for x, y in zip(c, a)]


def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class GateSizes:
    """Size model built on two structural facts found in main-round data (FINDINGS.md, "size gate"):
      1. Given a big win (>= 2x), the split among 2x/5x/20x is a fixed constant per machine type.
         -> learned in fit() as the pooled type-level split from train histories.
      2. Only a = P(1x | win) moves, and it follows a per-machine binary "size gate" that depends on what
         happened at one earlier lag (loss / 1x / big), with machine-specific lag and direction.
         -> Bayesian model averaging over hypotheses H_k: "a depends on the payout category at lag k",
            k in 1..max_lag, plus H_0: "a is constant". Each hypothesis gives every category cell its own
            Beta prior centred on the type mean (strength `strength`); cells include "missing" so every
            hypothesis scores the same wins. Weights = prior x Beta-binomial marginal likelihood.
    Nothing about which lag matters is hard-coded; hypothesis prior weights are learned from train
    (how often each lag wins on train machines), floored so a new round can pick different lags.
    `half_life` optionally down-weights old wins (applies to counts in every hypothesis).
    """
    def __init__(self, max_lag=8, strength=4.0, p_none=0.3, learn_prior=True, prior_floor=0.02, half_life=None,
                 cell_prior="beta", grid=49, npmle_iters=200, split_conc_cap=1e6,
                 couple=False, couple_alpha=5.0, couple_mix=0.1, couple_iters=1, win_cfg=None, tied=False, tie_weight=0.7, part_floor=0.02):
        self.tied, self.tie_weight, self.part_floor = tied, tie_weight, part_floor
        self.tie_weight_setting = tie_weight   # "learn" -> fitted on train in fit()
        self.levels = None; self.part_prior = None
        self.couple, self.couple_alpha, self.couple_mix = couple, couple_alpha, couple_mix
        self.couple_iters = couple_iters
        self.win_cfg = win_cfg or dict(a=.5, b=.5, p_change=.5, step=2, tau_min=40)
        self.max_lag, self.strength, self.p_none = max_lag, strength, p_none
        self.learn_prior, self.prior_floor, self.half_life = learn_prior, prior_floor, half_life
        self.cell_prior, self.grid, self.npmle_iters = cell_prior, grid, npmle_iters
        self.split_conc_cap = split_conc_cap
        self.atoms = None   # {type: (grid a values, log a, log(1-a), weights)} when cell_prior == "npmle"

    @staticmethod
    def _cat(v):
        return 3 if v < 0 else (0 if v == 0 else 1 if v == 1 else 2)   # loss, 1x, big, missing

    def _cell(self, n0, n1, t, mu):
        """(log marginal likelihood, posterior mean of a) for one cell with n1 1x-wins and n0 big wins."""
        if self.atoms is None or t not in self.atoms:
            a0, b0 = self.strength * mu, self.strength * (1 - mu)
            return _lbeta(a0 + n1, b0 + n0) - _lbeta(a0, b0), (a0 + n1) / (a0 + b0 + n0 + n1)
        A, la, lb, w = self.atoms[t]
        l = [n1 * x + n0 * y + math.log(wi) if wi > 0 else -1e300 for x, y, wi in zip(la, lb, w)]
        m = max(l); e = [math.exp(x - m) for x in l]; s = sum(e)
        return m + math.log(s), sum(ei * ai for ei, ai in zip(e, A)) / s

    def _cells(self, seq_m):
        """Per hypothesis: list of 4 cells [n0, n1] (index 0 = constant hypothesis, one pooled cell) and the
        index of the cell the cutoff falls in."""
        cut = len(seq_m); L = self.max_lag
        wins = [(t, int(seq_m[t] == 1)) for t in range(cut) if seq_m[t] > 0]
        wt = [1.0 if self.half_life is None else 2.0 ** (-(cut - 1 - t) / self.half_life) for t, _ in wins]
        n1 = sum(w * y for (t, y), w in zip(wins, wt)); n = sum(wt)
        out = [([[n - n1, n1]], 0)]
        for k in range(1, L + 1):
            c = [[0.0, 0.0] for _ in range(4)]
            for (t, y), w in zip(wins, wt):
                c[self._cat(seq_m[t - k]) if t - k >= 0 else 3][y] += w
            out.append((c, self._cat(seq_m[cut - k]) if cut - k >= 0 else 3))
        return out

    # Tied ("two-state") hypotheses. Main-round train: the categories (loss, 1x, big) at the gate lag split into
    # two groups that sit at the type's two learned levels (big vs not-big for nearly every machine). All three
    # two-group splits are allowed; their prior is learned in fit(). Missing lag -> state unknown (mixture).
    PARTITIONS = ((2,), (1,), (0,))   # group B = {big}, {1x}, {loss}; group A = the other two categories

    def _tied(self, cells, cur, t, k_is_none=False):
        """Combined (log marginal likelihood, predicted a) over tied hypotheses for one lag."""
        lo, hi = self.levels[t]; lvl = (lo, hi)
        def ll_cell(c0, c1, a):
            return c1 * math.log(a) + c0 * math.log(1 - a)
        def ll_mix(c0, c1):   # wins whose lag was missing: state unknown, each win a 50/50 mixture
            p1 = (lo + hi) / 2
            return c1 * math.log(p1) + c0 * math.log(1 - p1)
        out = []
        if k_is_none:   # constant state, lo or hi
            c0, c1 = cells[0]
            for s in (0, 1):
                out.append((math.log(.5) + ll_cell(c0, c1, lvl[s]), lvl[s]))
        else:
            for pi, grp in enumerate(self.PARTITIONS):
                lp = math.log(self.part_prior[pi]) + math.log(.5)
                for d in (0, 1):   # d = state of group B
                    ll = lp
                    for ci in range(3):
                        c0, c1 = cells[ci]
                        ll += ll_cell(c0, c1, lvl[d if ci in grp else 1 - d])
                    ll += ll_mix(*cells[3])
                    a = (lo + hi) / 2 if cur == 3 else lvl[d if cur in grp else 1 - d]
                    out.append((ll, a))
        m = max(x for x, _ in out); w = [math.exp(x - m) for x, _ in out]; z = sum(w)
        return m + math.log(z), sum(wi * a for wi, (_, a) in zip(w, out)) / z

    def _tables(self, seq_m, mu, t=None):
        """Per hypothesis (no gate, lag 1..max_lag): (log marginal likelihood, predicted a at the cutoff).
        With tying on, each lag mixes its free-cell model with its tied two-state model (weight tie_weight)."""
        res = []
        tie = self.tied and self.levels is not None and t in self.levels and isinstance(self.tie_weight, float)
        for k, (cells, cur) in enumerate(self._cells(seq_m)):
            parts = [self._cell(c0, c1, t, mu) for c0, c1 in cells]
            free = (sum(p[0] for p in parts), parts[cur][1])
            if not tie:
                res.append(free); continue
            tl = self._tied(cells, cur, t, k_is_none=(k == 0))
            w = self.tie_weight
            l1 = math.log(1 - w) + free[0]; l2 = math.log(w) + tl[0]; m = max(l1, l2)
            e1, e2 = math.exp(l1 - m), math.exp(l2 - m)
            res.append((m + math.log(e1 + e2), (e1 * free[1] + e2 * tl[1]) / (e1 + e2)))
        return res

    def _fit_atoms(self, queries, seq):
        """NPMLE (Kiefer-Wolfowitz, EM on a fixed grid) of the distribution of a per machine type, using the
        cells of each train machine's most likely gate hypothesis (from the Beta pass). Learns the state
        levels from data instead of hard-coding them."""
        G = [(i + 1) / (self.grid + 1) for i in range(self.grid)]
        la = [math.log(g) for g in G]; lb = [math.log(1 - g) for g in G]
        cells = defaultdict(list)
        for r in queries:
            s = seq[r["machine_id"]]; t = r["machine_type"]
            tabs = self._tables(s, self.mu[t]); hyp = self._cells(s)
            best = max(range(len(tabs)), key=lambda i: tabs[i][0])
            for c0, c1 in hyp[best][0]:
                if c0 + c1 >= 5:
                    cells[t].append((c0, c1))
        self.atoms = {}
        for t, cs in cells.items():
            E = []
            for c0, c1 in cs:
                row = [c1 * x + c0 * y for x, y in zip(la, lb)]; m = max(row)
                E.append([math.exp(v - m) for v in row])
            w = [1 / len(G)] * len(G)
            for _ in range(self.npmle_iters):
                acc = [0.0] * len(G)
                for e in E:
                    pe = [x * y for x, y in zip(e, w)]; s = sum(pe)
                    for j, v in enumerate(pe):
                        acc[j] += v / s
                w = [x / len(E) for x in acc]
            self.atoms[t] = (G, la, lb, w)

    def fit(self, queries, seq):
        big = defaultdict(lambda: [0, 0, 0]); ones = defaultdict(lambda: [0, 0])
        for r in queries:
            for v in seq[r["machine_id"]]:
                if v > 0:
                    ones[r["machine_type"]][int(v == 1)] += 1
                if v > 1:
                    big[r["machine_type"]][SIZES.index(v) - 1] += 1
        self.split = {t: [(x + .5) / (sum(c) + 1.5) for x in c] for t, c in big.items()}
        # Self-check of "big-win split is fixed per type": fit a Dirichlet-multinomial concentration per type
        # (Minka fixed point, mean held at the pooled split) on train machines' big-win counts. If the split
        # really is constant the concentration runs to the cap and predictions pool; if a new round makes it
        # vary by machine, the concentration drops and each machine's own counts take over.
        self.split_conc = {}
        for t, mean in self.split.items():
            cs = []
            for r in queries:
                if r["machine_type"] == t:
                    s = seq[r["machine_id"]]
                    c = [sum(1 for v in s if v == x) for x in (2, 5, 20)]
                    if sum(c) > 0:
                        cs.append(c)
            A = 10.0
            for _ in range(200):
                num = sum(m * (_digamma(c[i] + A * m) - _digamma(A * m)) for c in cs for i, m in enumerate(mean))
                den = sum(_digamma(sum(c) + A) - _digamma(A) for c in cs)
                A = min(self.split_conc_cap, A * num / den) if den > 0 else self.split_conc_cap
            self.split_conc[t] = A
        self.mu = {t: (c[1] + 1) / (sum(c) + 2) for t, c in ones.items()}
        self.atoms = None; self.levels = None
        if self.cell_prior == "npmle":
            self._fit_atoms(queries, seq)
        if self.tied and self.atoms is not None:
            self._fit_levels_and_partitions(queries, seq)
            if self.tie_weight_setting == "learn":
                self._fit_tie_weight(queries, seq)
        L = self.max_lag
        prior = [self.p_none] + [(1 - self.p_none) / L] * L
        if self.learn_prior:   # how often each hypothesis is the most likely one on train machines
            score = [0.0] * (L + 1)
            for r in queries:
                tabs = self._tables(seq[r["machine_id"]], self.mu[r["machine_type"]], r["machine_type"])
                lls = [x[0] for x in tabs]; m = max(lls); e = [math.exp(x - m) for x in lls]; s = sum(e)
                for i, x in enumerate(e):
                    score[i] += x / s
            tot = sum(score); prior = [max(self.prior_floor, x / tot) for x in score]
            s = sum(prior); prior = [x / s for x in prior]
        self.prior = prior
        if self.couple:
            self._fit_coupling(queries, seq)
        return self

    def _fit_tie_weight(self, queries, seq, iters=20):
        """EM for the weight of the two-state (tied) model vs free cells, on train machines' best lag.
        If a new round's sizes are not two-state, tied hypotheses lose likelihood and the weight falls,
        so the free-cell fallback takes over automatically. Clipped to [0.02, 0.98]."""
        pairs = []
        for r in queries:
            s = seq[r["machine_id"]]; t = r["machine_type"]
            if t not in self.levels:
                continue
            hyp = self._cells(s)
            free = [sum(self._cell(c0, c1, t, self.mu[t])[0] for c0, c1 in cells) for cells, _ in hyp]
            k = max(range(1, len(hyp)), key=free.__getitem__)
            cells, cur = hyp[k]
            pairs.append((free[k], self._tied(cells, cur, t)[0]))
        w = 0.5
        for _ in range(iters):
            acc = 0.0
            for lf, lt in pairs:
                a, b = math.log(1 - w) + lf, math.log(w) + lt; m = max(a, b)
                acc += math.exp(b - m) / (math.exp(a - m) + math.exp(b - m))
            w = min(.98, max(.02, acc / max(1, len(pairs))))
        self.tie_weight = w

    def _fit_levels_and_partitions(self, queries, seq):
        """Two levels per type = weighted means of the learned NPMLE atoms below / above the type mean.
        Partition prior = average posterior over the 3 two-group splits, on each train machine's best free lag."""
        self.levels = {}
        for t, (A, _, _, w) in self.atoms.items():
            mu = self.mu[t]
            lo = [(a, x) for a, x in zip(A, w) if a < mu]; hi = [(a, x) for a, x in zip(A, w) if a >= mu]
            if not lo or not hi or sum(x for _, x in lo) < .05 or sum(x for _, x in hi) < .05:
                continue   # not two-state for this type: tied hypotheses switched off for it
            self.levels[t] = (sum(a * x for a, x in lo) / sum(x for _, x in lo),
                              sum(a * x for a, x in hi) / sum(x for _, x in hi))
        self.part_prior = [1 / len(self.PARTITIONS)] * len(self.PARTITIONS)
        acc = [0.0] * len(self.PARTITIONS)
        for r in queries:
            s = seq[r["machine_id"]]; t = r["machine_type"]
            if t not in self.levels:
                continue
            hyp = self._cells(s); lv = self.levels
            free = [sum(self._cell(c0, c1, t, self.mu[t])[0] for c0, c1 in cells) for cells, _ in hyp]
            k = max(range(1, len(hyp)), key=free.__getitem__)
            cells, cur = hyp[k]; lo, hi = lv[t]
            lls = []
            for grp in self.PARTITIONS:
                best = []
                for d in (0, 1):
                    best.append(sum(cells[ci][1] * math.log((lo, hi)[d if ci in grp else 1 - d]) +
                                    cells[ci][0] * math.log(1 - (lo, hi)[d if ci in grp else 1 - d]) for ci in range(3)))
                m = max(best); lls.append(m + math.log(sum(math.exp(b - m) for b in best)))
            m = max(lls); e = [math.exp(x - m) for x in lls]; z = sum(e)
            for i, x in enumerate(e):
                acc[i] += x / z
        z = sum(acc)
        if z > 0:
            p = [max(self.part_floor, x / z) for x in acc]; z = sum(p); self.part_prior = [x / z for x in p]

    # ---- Coupling to the win gate (short-history fix; FINDINGS.md "size lag follows the win gate") ----
    # The size-gate lag is tied to the machine's win-gate structure (main round: win pair (a, b) -> size lag a).
    # Wins identify the win gate from every observed tick, so P(win structure | history) is a much sharper
    # prior for the size lag than the global one. The table P(size lag | win structure) is learned in fit().
    def _win_posterior(self, seq_m):
        """Posterior over win-gate structures (method 1's family + 'no gate'), flips marginalized like method 1."""
        cut = len(seq_m); cfg = self.win_cfg
        taus = list(range(cfg["tau_min"], cut - 2, cfg["step"]))
        lt = [math.log(cfg["p_change"] / len(taus))] * len(taus) + [math.log(1 - cfg["p_change"])] if taus \
            else [0.0]
        tb = machine_tables(seq_m, cut, taus, self.wST, cfg["a"], cfg["b"])
        ls = []
        for si, rows in enumerate(tb):
            ls.append(math.log(self.wprior[si]) + _logsumexp([ll + p for (ll, _), p in zip(rows, lt)]))
        m = max(ls); e = [math.exp(x - m) for x in ls]; z = sum(e)
        return [x / z for x in e]

    def _size_posterior(self, tabs, prior):
        lw = [math.log(max(p, 1e-300)) + ll for p, (ll, _) in zip(prior, tabs)]; m = max(lw)
        w = [math.exp(x - m) for x in lw]; s = sum(w)
        return [x / s for x in w]

    def _fit_coupling(self, queries, seq):
        self.wST = structures(self.max_lag, 2)
        self.wprior = learn_structure_prior(queries, seq, self.wST, self.win_cfg["a"], self.win_cfg["b"])
        K = self.max_lag + 1; S = len(self.wST) + 1; a = self.couple_alpha
        data = []
        for r in queries:
            s = seq[r["machine_id"]]; t = r["machine_type"]
            wp = self._win_posterior(s); ll = [x[0] for x in self._tables(s, self.mu[t], t)]
            m = max(ll); L = [math.exp(x - m) for x in ll]
            data.append(([(i, x) for i, x in enumerate(wp) if x > 1e-4], L))
        self.cond = [self.prior[:] for _ in range(S)]
        # EM over the joint latent (win structure i, size lag k): P(i, k | data) ∝ wp_i * cond[i][k] * L_k
        for _ in range(self.couple_iters):
            C = [[0.0] * K for _ in range(S)]
            for wp, L in data:
                joint = [(i, k, x * self.cond[i][k] * L[k]) for i, x in wp for k in range(K)]
                z = sum(v for _, _, v in joint)
                for i, k, v in joint:
                    C[i][k] += v / z
            self.cond = [[(C[i][k] + a * self.prior[k]) / (sum(C[i]) + a) for k in range(K)] for i in range(S)]

    def size_prior(self, seq_m):
        if not self.couple:
            return self.prior
        wp = self._win_posterior(seq_m); K = self.max_lag + 1
        p = [sum(wp[i] * self.cond[i][k] for i in range(len(wp))) for k in range(K)]
        m = self.couple_mix   # keep some global prior so a wrong coupling can't zero out a hypothesis
        return [(1 - m) * x + m * y for x, y in zip(p, self.prior)]

    def predict(self, q, seq_m):
        t = q["machine_type"]; tabs = self._tables(seq_m, self.mu[t], t)
        lw = [math.log(p) + ll for p, (ll, _) in zip(self.size_prior(seq_m), tabs)]; m = max(lw)
        w = [math.exp(x - m) for x in lw]; s = sum(w)
        a = sum(wi * pa for wi, (_, pa) in zip(w, tabs)) / s
        A = self.split_conc[t]; c = [sum(1 for v in seq_m if v == x) for x in (2, 5, 20)]
        sp = [(ci + A * m) / (sum(c) + A) for ci, m in zip(c, self.split[t])]
        return [a] + [(1 - a) * x for x in sp]


class LagSizes:
    """Size given a win, conditioned on whether earlier spins won (B1 finding: per-machine, direction varies).

    For each lag k in `lags`, the covariate is win/loss at tick (cutoff - k). Counts of this machine's sizes in
    past ticks with the same covariate pattern are shrunk toward `base`'s prediction for the machine:
        p = (counts_same_cell + strength * base) / (n_cell + strength)
    With two lags the cells nest: the joint cell shrinks toward the lag-1-only cell, which shrinks toward base.
    `half_life` (ticks) optionally down-weights old wins. Missing ticks never form a cell.
    """
    def __init__(self, base, lags=(1,), strength=20.0, half_life=None):
        self.base, self.lags, self.strength, self.half_life = base, tuple(lags), strength, half_life

    def fit(self, queries, seq):
        self.base.fit(queries, seq)
        return self

    def _cell_probs(self, seq_m, lags, prior):
        cut = len(seq_m)
        cur = [seq_m[cut - k] for k in lags]
        if any(v < 0 for v in cur):
            return prior
        want = tuple(int(v > 0) for v in cur)
        c = [0.0] * 4
        for t in range(max(lags), cut):
            v = seq_m[t]
            if v <= 0:
                continue
            past = [seq_m[t - k] for k in lags]
            if any(x < 0 for x in past) or tuple(int(x > 0) for x in past) != want:
                continue
            w = 1.0 if self.half_life is None else 2.0 ** (-(cut - 1 - t) / self.half_life)
            c[SIZES.index(v)] += w
        n = sum(c)
        return [(x + self.strength * p) / (n + self.strength) for x, p in zip(c, prior)]

    def predict(self, q, seq_m):
        p = self.base.predict(q, seq_m)
        for i in range(1, len(self.lags) + 1):
            p = self._cell_probs(seq_m, self.lags[:i], p)
        return p


class NPMLESizes:
    """Kiefer-Wolfowitz NPMLE prior over machine size distributions, per machine type (Koenker & Gu).
    Atoms: smoothed size distributions of train machines with >= min_wins wins. Weights: EM on train.
    Prediction: posterior mean over atoms given the machine's own size counts."""
    def __init__(self, min_wins=30, iters=300, half_life=None):
        self.min_wins, self.iters, self.half_life = min_wins, iters, half_life

    def fit(self, queries, seq):
        atoms = defaultdict(list); C = defaultdict(list)
        for r in queries:
            c = size_counts(seq[r["machine_id"]]); C[r["machine_type"]].append(c)
            if sum(c) >= self.min_wins:
                atoms[r["machine_type"]].append([(x + .5) / (sum(c) + 2) for x in c])
        self.model = {}
        for t, A in atoms.items():
            logA = [[math.log(v) for v in a] for a in A]
            E = []
            for c in C[t]:
                row = [sum(c[i] * la[i] for i in range(4)) for la in logA]; m = max(row)
                E.append([math.exp(x - m) for x in row])
            w = [1 / len(A)] * len(A)
            for _ in range(self.iters):
                acc = [0.0] * len(A)
                for e in E:
                    pe = [x * y for x, y in zip(e, w)]; s = sum(pe)
                    for j, v in enumerate(pe):
                        acc[j] += v / s
                w = [a / len(E) for a in acc]
            self.model[t] = (A, logA, w)
        return self

    def predict(self, q, seq_m):
        A, logA, wt = self.model[q["machine_type"]]
        if self.half_life is None:
            c = size_counts(seq_m)
        else:  # B4: recency-weighted counts in the posterior (atoms/weights still fit on full train counts)
            cut = len(seq_m); c = [0.0] * 4
            for t, v in enumerate(seq_m):
                if v > 0:
                    c[SIZES.index(v)] += 2.0 ** (-(cut - 1 - t) / self.half_life)
        l = [sum(c[i] * la[i] for i in range(4)) + (math.log(x) if x > 0 else -1e300) for la, x in zip(logA, wt)]
        m = max(l); p = [math.exp(x - m) for x in l]; s = sum(p)
        return [sum(p[j] * A[j][i] for j in range(len(A))) / s for i in range(4)]
