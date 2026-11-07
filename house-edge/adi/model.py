"""adi: Method 3 (structure discovery) model. Method 1's change-point gate over a wider, learned structure family.

Settings
  max_lag      lags considered for XOR structures (default 8 = method 1)
  xor_order    1 singles, 2 + pairs (default), 3 + triples
  and_pairs    also include AND-family pairs (AND/OR/NAND/NOR with input negations)
  and_max_lag  lag range for AND pairs (default = max_lag)
  prior_flips  learn the structure prior with flip times marginalized (True) or ignored (False, = method 1)
  prune        drop structures whose learned prior weight is below this, then re-normalize
  floor        uniform mixing weight over the kept structures (method 1 uses 0.1)
  none_floor   minimum prior weight on 'no gate'
  cell_prior   None (Beta(a, b) per gate cell, = method 1), "learn" (C5: asymmetric on/off Beta priors fitted on
               train by marginal likelihood), or an explicit [a_on, b_on, a_off, b_off]
  cell_group   with cell_prior="learn": "none" (one shared prior), "type", or "type_signal" (default; per machine
               type x signal, falling back to type / shared for groups with < 40 train machines)
  relearn      after fitting the cell prior, re-learn the structure prior with the shared cell prior (C5)
  p_change_short  flip-prior weight used for short logs (more than half the ticks missing); None = p_change,
               "learn" = pick from P_CHANGE_GRID by marginal likelihood of the short train logs (C5)
  episode_shift  None (off), "both" / "on" / "off" / "contrast", or "learn" (pick the mode by train evidence).
               Each episode gets a shared log-odds shift delta of its machines' on/off cell-prior means,
               delta ~ discretized N(0, shift_sigma^2) on SHIFT_GRID. P(win) for each machine is averaged over the
               posterior of delta given every machine's log in the episode (all ticks before the cutoff).
  shift_sigma  float, or "learn" (pick from SIGMA_GRID by train marginal likelihood over episodes; 0 = no shift)
  rate_half_life  None, ticks, or {machine_type: ticks}: down-weight old ticks in the gate-cell counts (rates drift slowly). Applied
               at prediction time to long logs only (it hurts short logs); priors are learned unweighted.
  neighbor_adjust  add beta * (episode neighbours' recent-minus-long-run win rate over the last W ticks) to
               P(win); W and beta fitted by least squares on train labels ("hot room" effect)
  size_split   True / "small" / "both" / "tied" / "tied+free": add "xs" structures, each kept XOR pair (i, j) with its cells split by
               the payout category (loss / 1x / big) at its smaller lag i ("both": also at j), so the win rate can
               depend on the hidden size state. "tied": cells merge into two size states (one category vs the
               other two, averaged over the 3 groupings) = 4 cells. Prior over plain + split re-learned under the
               cell priors (round 4)
  cell_signal  with cell_prior="learn": per machine type, on/off prior means get a learned log-odds slope in
               `signal` (hierarchical; strengths shared within the type), fitted on train evidence (round 5)
  split_long_only  with size_split: use split structures only for long logs (short logs keep plain pairs)
  sizes        "npmle" (default, = method 1), "dirichlet", or "gate" (method2_sizes.GateSizes, imported as-is)
  gate_*       passed to GateSizes when sizes="gate": gate_max_lag, gate_strength, gate_p_none,
               gate_learn_prior, gate_half_life, gate_cell_prior, gate_couple, gate_couple_alpha, gate_couple_mix,
               gate_couple_iters, gate_tied, gate_tie_weight (Person B's best: gate_cell_prior="npmle",
               gate_couple=True, gate_couple_iters=20, gate_tied=True). GateSizes' coupling uses its own
               method-1 win-gate posterior with this model's a/b/p_change/step/tau_min.
Everything is learned in fit() from pub/train only. Probabilities = [1 - P(win), P(win) * size distribution].
"""
import csv, math, os
from common.data import load_split
from method2_sizes.sizes import NPMLESizes, DirichletSizes, GateSizes  # read-only import of Person B's models
from method1_gate.gate import learn_structure_prior as m1_prior
from adi import gate as G


P_CHANGE_GRID = (.05, .1, .2, .3, .4, .5, .7)
SHIFT_GRID = (-1.2, -.9, -.6, -.3, 0.0, .3, .6, .9, 1.2)
SIGMA_GRID = (0.0, .1, .2, .3, .45, .6, .8)
SHIFT_MODES = ("both", "on", "off", "contrast")
NBR_WINDOWS = (5, 10, 20)


def neighbor_feature(q, seq, W):
    """Per query: mean over the other machines in its episode of (win rate over the W ticks before the cutoff
    minus their win rate before that). Uses only ticks strictly before the cutoff."""
    eps = {}
    for r in q:
        eps.setdefault(r["episode_id"], []).append(r)
    out = {}
    for r in q:
        cut = r["cutoff"]; vals = []
        for o in eps[r["episode_id"]]:
            if o["machine_id"] == r["machine_id"]:
                continue
            so = seq[o["machine_id"]]
            rec = [x for x in so[cut - W:cut] if x >= 0]; old = [x for x in so[:cut - W] if x >= 0]
            if rec and old:
                vals.append(sum(x > 0 for x in rec) / len(rec) - sum(x > 0 for x in old) / len(old))
        out[r["query_id"]] = sum(vals) / len(vals) if vals else 0.0
    return out


def shift_logprior(sigma):
    if sigma <= 0:
        return [0.0 if d == 0 else -1e300 for d in SHIFT_GRID]
    l = [-d * d / (2 * sigma * sigma) for d in SHIFT_GRID]; z = G.logsumexp(l)
    return [x - z for x in l]


def is_short(s):
    return sum(x < 0 for x in s) > len(s) / 2


class Model:
    def __init__(self, max_lag=8, xor_order=2, and_pairs=False, and_max_lag=None, prior_flips=False,
                 prune=0.0, floor=0.1, none_floor=0.05, a=.5, b=.5, p_change=.5, step=2, tau_min=40,
                 cell_prior=None, cell_group="type_signal", relearn=False,
                 p_change_short=None, sizes="npmle", size_strength=30.0, gate_max_lag=8, gate_strength=4.0,
                 gate_p_none=0.3, gate_learn_prior=True, gate_half_life=None, gate_cell_prior="beta",
                 gate_couple=False, gate_couple_alpha=5.0, gate_couple_mix=0.1, gate_couple_iters=1,
                 gate_tied=False, gate_tie_weight=0.7, episode_shift=None, shift_sigma="learn",
                 rate_half_life=None, neighbor_adjust=False, size_split=False,
                 split_long_only=False, cell_signal=False):
        self.cfg = dict(a=a, b=b, p_change=p_change, step=step, tau_min=tau_min, half_life=None)
        self.rate_half_life = rate_half_life
        self.ST = G.structures(max_lag, xor_order, and_pairs, and_max_lag)
        self.prior_flips, self.prune, self.floor, self.none_floor = prior_flips, prune, floor, none_floor
        self.cell_prior, self.relearn, self.p_change_short = cell_prior, relearn, p_change_short
        self.cell_group = cell_group
        self.episode_shift, self.shift_sigma = episode_shift, shift_sigma
        self.neighbor_adjust, self.size_split = neighbor_adjust, size_split
        self.split_long_only, self.cell_signal = split_long_only, cell_signal
        self.cellpriors = None if cell_prior in (None, "learn") else {(): tuple(cell_prior)}
        if sizes == "gate":
            self.sizes = GateSizes(max_lag=gate_max_lag, strength=gate_strength, p_none=gate_p_none,
                                   learn_prior=gate_learn_prior, half_life=gate_half_life, cell_prior=gate_cell_prior,
                                   couple=gate_couple, couple_alpha=gate_couple_alpha, couple_mix=gate_couple_mix,
                                   couple_iters=gate_couple_iters, win_cfg=self.cfg, tied=gate_tied,
                                   tie_weight=gate_tie_weight)
        elif sizes == "dirichlet":
            self.sizes = DirichletSizes(size_strength)
        elif sizes == "npmle":
            self.sizes = NPMLESizes()
        else:
            raise ValueError(f"unknown sizes option {sizes!r}")

    def _raw_prior(self, q, seq):
        if self.prior_flips:
            return G.learn_structure_prior(q, seq, self.ST, self.cfg["a"], self.cfg["b"], self.cfg["p_change"],
                                           tau_min=self.cfg["tau_min"])
        if all(f == "xor" for f, _ in self.ST):  # identical to method 1's prior (before its floors)
            return m1_prior(q, seq, [s for _, s in self.ST], self.cfg["a"], self.cfg["b"], none_floor=0, uniform_mix=0)
        return G.learn_structure_prior(q, seq, self.ST, self.cfg["a"], self.cfg["b"], p_change=1e-12,
                                       tau_min=self.cfg["tau_min"])

    def _finish_prior(self, p):
        s = sum(p); p = [(1 - self.floor) * x / s + self.floor / len(p) for x in p]
        p[-1] = max(p[-1], self.none_floor); s = sum(p)
        return [x / s for x in p]

    def fit(self, pub):
        q, seq = load_split(pub, "train")
        p = self._raw_prior(q, seq)  # len(ST) + 1, last = 'no gate'
        if self.prune > 0:
            keep = [i for i in range(len(self.ST)) if p[i] >= self.prune]
            self.ST = [self.ST[i] for i in keep]; p = [p[i] for i in keep] + [p[-1]]
        self.sprior = self._finish_prior(p)
        if self.cell_prior == "learn":
            self.cellpriors = G.fit_cell_priors(q, seq, self.ST, self.sprior, self.cfg, self.cell_group)
        if self.cell_signal and self.cell_prior == "learn":
            # round 5: per machine type, means shifted by a learned log-odds slope in signal (start from the
            # type prior already fitted; fitted on train evidence)
            self.signal_priors = {}
            for t in sorted({r["machine_type"] for r in q}):
                rs = [r for r in q if r["machine_type"] == t]
                self.signal_priors[t] = G.fit_signal_cell_prior(rs, seq, self.ST, self.sprior, self.cfg)[0]
        if self.size_split:
            self._add_splits(q, seq)
        if self.relearn and self.cellpriors is not None:
            p = G.learn_structure_prior(q, seq, self.ST, self.cfg["a"], self.cfg["b"], self.cfg["p_change"],
                                        tau_min=self.cfg["tau_min"], cellprior=self.cellpriors[()])
            self.sprior = self._finish_prior(p)
        if self.p_change_short == "learn":
            short = [r for r in q if is_short(seq[r["machine_id"]])]
            def ev(pc):
                cfg = dict(self.cfg, p_change=pc, step=max(4, self.cfg["step"]))
                return sum(G.win_probability(r, seq[r["machine_id"]], self.ST, self.sprior, evidence=True,
                                             cellprior=self._cp(r), **cfg)[1] for r in short)
            self.p_change_short = max(P_CHANGE_GRID, key=ev)
        if self.episode_shift is not None:
            self._fit_shift(q, seq)
        if self.neighbor_adjust:
            self._fit_neighbor(pub, q, seq)
        self.sizes.fit(q, seq)
        return self

    def _add_splits(self, q, seq):
        """Add size-split variants ("xs") of every kept XOR pair and re-learn the structure prior over plain +
        split structures under the learned per-type cell priors (flip times marginalized, train only)."""
        pairs = [st for f, st in self.ST if f == "xor" and len(st) == 2]
        mode = self.size_split if isinstance(self.size_split, str) else "small"
        if mode in ("small", "both"):
            ks = (lambda i, j: (i,)) if mode == "small" else (lambda i, j: (i, j))
            self.ST = self.ST + [("xs", (i, j, k)) for i, j in pairs for k in ks(i, j)]
        if mode in ("tied", "tied+free"):
            self.ST = self.ST + [("xst", (i, j, i, c)) for i, j in pairs for c in range(3)]
        if mode == "tied+free":
            self.ST = self.ST + [("xs", (i, j, i)) for i, j in pairs]
        acc = [0.0] * (len(self.ST) + 1)
        for r in q:
            cfg = self._mcfg(seq[r["machine_id"]], step=4, r_type=r["machine_type"])
            taus, lp = G._taus(r["cutoff"], cfg["p_change"], cfg["step"], cfg["tau_min"])
            tb = G.machine_tables(seq[r["machine_id"]], r["cutoff"], taus, self.ST, cfg["a"], cfg["b"],
                                  self._cp(r), cfg["half_life"])
            l = [G.logsumexp([ll + x for (ll, _), x in zip(rows, lp)]) for rows in tb]; z = G.logsumexp(l)
            for i, x in enumerate(l):
                acc[i] += math.exp(x - z)
        self.split_raw_prior = [x / sum(acc) for x in acc]
        self.sprior = self._finish_prior(list(self.split_raw_prior))

    def _sp(self, s):
        """Structure prior for this machine: split structures dropped for short logs if split_long_only."""
        if not (self.size_split and self.split_long_only and is_short(s)):
            return self.sprior
        p = [0.0 if i < len(self.ST) and self.ST[i][0] in ("xs", "xst") else x for i, x in enumerate(self.sprior)]
        z = sum(p); return [x / z for x in p]

    def _mcfg(self, s, step=None, r_type=None):
        cfg = dict(self.cfg)
        if isinstance(self.p_change_short, float) and is_short(s):
            cfg["p_change"] = self.p_change_short
        if step is not None:
            cfg["step"] = max(step, cfg["step"])
        if self.rate_half_life is not None and not is_short(s):
            hl = self.rate_half_life
            cfg["half_life"] = hl.get(r_type, None) if isinstance(hl, dict) else hl
        return cfg

    def _grid(self, r, s, mode, step=None):
        cp = self._cp(r)
        cps = [G.shift_cellprior(cp, d, mode) for d in SHIFT_GRID]
        return G.win_probability_grid(r, s, self.ST, self._sp(s), cps, **self._mcfg(s, step, r["machine_type"]))

    def _fit_shift(self, q, seq):
        """Pick (mode, sigma) maximizing sum over train episodes of log sum_delta prior(delta) prod_m evidence_m."""
        modes = SHIFT_MODES if self.episode_shift == "learn" else (self.episode_shift,)
        sigmas = SIGMA_GRID if self.shift_sigma == "learn" else (float(self.shift_sigma),)
        eps = {}
        for r in q:
            eps.setdefault(r["episode_id"], []).append(r)
        best = None; self.shift_table = {}
        for mode in modes:
            ev = {e: [0.0] * len(SHIFT_GRID) for e in eps}
            for e, rs in eps.items():
                for r in rs:
                    g = self._grid(r, seq[r["machine_id"]], mode, step=4)
                    ev[e] = [x + z for x, (_, z) in zip(ev[e], g)]
            for sg in sigmas:
                lp = shift_logprior(sg)
                tot = sum(G.logsumexp([a + b for a, b in zip(lp, v)]) for v in ev.values())
                self.shift_table[(mode, sg)] = tot
                if best is None or tot > best[0]:
                    best = (tot, mode, sg)
        _, self.shift_mode, self.shift_sigma_fit = best

    def _cp(self, r):
        if getattr(self, "signal_priors", None) and r["machine_type"] in self.signal_priors:
            return G.signal_cellprior(self.signal_priors[r["machine_type"]], r["signal"])
        if self.cellpriors is None:
            return None
        return G.lookup_cell_prior(self.cellpriors, r, self.cell_group if self.cell_prior == "learn" else "none")

    def win_probs(self, q, seq):
        """{query_id: P(win)} from the gate (with the episode shift if enabled), before any neighbour adjustment."""
        if self.episode_shift is not None and self.shift_sigma_fit > 0:
            lp = shift_logprior(self.shift_sigma_fit); eps = {}; out = {}
            for r in q:
                eps.setdefault(r["episode_id"], []).append(r)
            for rs in eps.values():
                grids = [self._grid(r, seq[r["machine_id"]], self.shift_mode) for r in rs]
                post = list(lp)
                for g in grids:
                    post = [x + z for x, (_, z) in zip(post, g)]
                zz = G.logsumexp(post); post = [math.exp(x - zz) for x in post]
                for r, g in zip(rs, grids):
                    out[r["query_id"]] = sum(w * p for w, (p, _) in zip(post, g))
            return out
        return {r["query_id"]: G.win_probability(r, seq[r["machine_id"]], self.ST, self._sp(seq[r["machine_id"]]),
                                                 cellprior=self._cp(r),
                                                 **self._mcfg(seq[r["machine_id"]], r_type=r["machine_type"]))
                for r in q}

    def predict(self, pub, split):
        q, seq = load_split(pub, split)
        pw = self.win_probs(q, seq)
        if self.neighbor_adjust:
            f = neighbor_feature(q, seq, self.nbr_window)
            pw = {k: min(1 - 1e-4, max(1e-4, v + self.nbr_beta * f[k])) for k, v in pw.items()}
        return {r["query_id"]: [1 - pw[r["query_id"]]] + [pw[r["query_id"]] * x for x in
                                                          self.sizes.predict(r, seq[r["machine_id"]])] for r in q}

    def _fit_neighbor(self, pub, q, seq):
        """Least squares (= the contest's squared error on P(win)) of train label minus P(win) on the neighbour
        feature, for each window in NBR_WINDOWS; keep the window with the largest error reduction."""
        with open(os.path.join(pub, "train", "labels.csv"), newline="") as fh:
            y = {r["query_id"]: float(int(r["outcome"]) > 0) for r in csv.DictReader(fh)}
        pw = self.win_probs(q, seq); res = {k: y[k] - v for k, v in pw.items()}
        best = (0.0, NBR_WINDOWS[0], 0.0)
        for W in NBR_WINDOWS:
            f = neighbor_feature(q, seq, W); ks = list(res)
            sxx = sum(f[k] ** 2 for k in ks); sxy = sum(f[k] * res[k] for k in ks)
            if sxx > 0 and sxy * sxy / sxx > best[0]:
                best = (sxy * sxy / sxx, W, sxy / sxx)
        _, self.nbr_window, self.nbr_beta = best
