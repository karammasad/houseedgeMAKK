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
Everything is learned in fit() from pub/train only. Probabilities = [1 - P(win), P(win) * size distribution].
"""
from common.data import load_split
from method2_sizes.sizes import NPMLESizes, DirichletSizes
from method1_gate.gate import learn_structure_prior as m1_prior
from adi import gate as G


P_CHANGE_GRID = (.05, .1, .2, .3, .4, .5, .7)


def is_short(s):
    return sum(x < 0 for x in s) > len(s) / 2


class Model:
    def __init__(self, max_lag=8, xor_order=2, and_pairs=False, and_max_lag=None, prior_flips=False,
                 prune=0.0, floor=0.1, none_floor=0.05, a=.5, b=.5, p_change=.5, step=2, tau_min=40,
                 cell_prior=None, cell_group="type_signal", relearn=False,
                 p_change_short=None, sizes="npmle", size_strength=30.0):
        self.cfg = dict(a=a, b=b, p_change=p_change, step=step, tau_min=tau_min)
        self.ST = G.structures(max_lag, xor_order, and_pairs, and_max_lag)
        self.prior_flips, self.prune, self.floor, self.none_floor = prior_flips, prune, floor, none_floor
        self.cell_prior, self.relearn, self.p_change_short = cell_prior, relearn, p_change_short
        self.cell_group = cell_group
        self.cellpriors = None if cell_prior in (None, "learn") else {(): tuple(cell_prior)}
        self.sizes = NPMLESizes() if sizes == "npmle" else DirichletSizes(size_strength)

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
        self.sizes.fit(q, seq)
        return self

    def _cp(self, r):
        if self.cellpriors is None:
            return None
        return G.lookup_cell_prior(self.cellpriors, r, self.cell_group if self.cell_prior == "learn" else "none")

    def predict(self, pub, split):
        q, seq = load_split(pub, split); out = {}
        for r in q:
            s = seq[r["machine_id"]]
            cfg = self.cfg
            if self.p_change_short is not None and is_short(s):
                cfg = dict(cfg, p_change=self.p_change_short)
            pw = G.win_probability(r, s, self.ST, self.sprior, cellprior=self._cp(r), **cfg)
            out[r["query_id"]] = [1 - pw] + [pw * x for x in self.sizes.predict(r, s)]
        return out
