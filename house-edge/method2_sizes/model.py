"""Method 2 model: Method 1's gate with a size model chosen here. Person B edits sizes.py and this file.

Size options (all standard library):
  sizes="npmle"      NPMLE prior per machine type (default, = current entry); npmle_half_life
                     recency-weights the machine's own counts (B4), npmle_min_wins sets atom threshold (B5)
  sizes="dirichlet"  Dirichlet shrinkage toward type x signal average (size_strength)
  sizes="minka"      B6 control: Dirichlet-multinomial per type, alpha by Minka fixed point
  sizes="lag"        LagSizes on top of NPMLE: size counts conditioned on win/loss at size_lags,
                     shrunk toward the NPMLE prediction with lag_strength; optional size_half_life.
  sizes="gate"       GateSizes: fixed per-type big-win split + Bayesian model averaging over a per-machine
                     "size gate" at lag 1..gate_max_lag (gate_strength, gate_p_none, gate_half_life);
                     gate_cell_prior="npmle" learns the distribution of P(1x|win) per type from train;
                     gate_couple=True sets the size-lag prior from the machine's win-gate posterior through a
                     table P(size lag | win structure) learned on train (short-history fix);
                     gate_tied=True adds two-state hypotheses (category groups at the two learned levels).
The gate (win probability) is untouched, so size changes are measured on their own.
"""
import importlib
from common.data import load_split
from method1_gate.model import Model as GateModel
from method2_sizes.sizes import NPMLESizes, DirichletSizes, LagSizes, MinkaSizes, GateSizes


class Model(GateModel):
    """win_model: None = method 1's gate (default). Or the package name of another team method whose Model
    gives the win probability (P(win) = 1 - p0 of its prediction), e.g. "adi" with win_kwargs = its settings.
    Sizes always come from this package's size model ("stacking": their gate x our size split)."""
    def fit(self, pub):
        if self.win_model:
            self.win = importlib.import_module(self.win_model + ".model").Model(**self.win_kwargs).fit(pub)
            q, seq = load_split(pub, "train"); self.sizes.fit(q, seq)
            return self
        return super().fit(pub)

    def predict(self, pub, split):
        if not self.win_model:
            return super().predict(pub, split)
        pw_pred = self.win.predict(pub, split)
        q, seq = load_split(pub, split); out = {}
        for r in q:
            p = pw_pred[r["query_id"]]; pw = 1 - p[0] / sum(p)
            out[r["query_id"]] = [1 - pw] + [pw * x for x in self.sizes.predict(r, seq[r["machine_id"]])]
        return out

    def __init__(self, sizes="npmle", size_lags=(1,), lag_strength=20.0, size_half_life=None,
                 npmle_min_wins=30, npmle_half_life=None,
                 gate_max_lag=8, gate_strength=4.0, gate_p_none=0.3, gate_learn_prior=True, gate_half_life=None, gate_cell_prior="beta",
                 gate_couple=False, gate_couple_alpha=5.0, gate_couple_mix=0.1, gate_couple_iters=1,
                 gate_tied=False, gate_tie_weight=0.7,
                 win_model=None, win_kwargs=None, **kw):
        self.win_model, self.win_kwargs = win_model, dict(win_kwargs or {})
        base_kind = {"dirichlet": "dirichlet", "minka": "minka"}.get(sizes, "npmle")
        super().__init__(sizes="dirichlet" if base_kind == "dirichlet" else "npmle", **kw)
        if base_kind == "minka":
            self.sizes = MinkaSizes()
        elif base_kind == "npmle":
            self.sizes = NPMLESizes(min_wins=npmle_min_wins, half_life=npmle_half_life)
        if sizes == "lag":
            self.sizes = LagSizes(self.sizes, lags=size_lags, strength=lag_strength, half_life=size_half_life)
        elif sizes == "gate":
            self.sizes = GateSizes(max_lag=gate_max_lag, strength=gate_strength, p_none=gate_p_none,
                                   learn_prior=gate_learn_prior, half_life=gate_half_life,
                                   cell_prior=gate_cell_prior, couple=gate_couple, couple_alpha=gate_couple_alpha,
                                   couple_mix=gate_couple_mix, couple_iters=gate_couple_iters, win_cfg=self.cfg,
                                   tied=gate_tied, tie_weight=gate_tie_weight)
        elif sizes not in ("npmle", "dirichlet", "minka"):
            raise ValueError(f"unknown sizes option {sizes!r}")
