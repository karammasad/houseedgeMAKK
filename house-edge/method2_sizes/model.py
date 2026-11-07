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
from method1_gate.model import Model as GateModel
from method2_sizes.sizes import NPMLESizes, DirichletSizes, LagSizes, MinkaSizes, GateSizes


class Model(GateModel):
    def __init__(self, sizes="npmle", size_lags=(1,), lag_strength=20.0, size_half_life=None,
                 npmle_min_wins=30, npmle_half_life=None,
                 gate_max_lag=8, gate_strength=4.0, gate_p_none=0.3, gate_learn_prior=True, gate_half_life=None, gate_cell_prior="beta",
                 gate_couple=False, gate_couple_alpha=5.0, gate_couple_mix=0.1, gate_couple_iters=1,
                 gate_tied=False, gate_tie_weight=0.7, **kw):
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
