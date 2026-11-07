"""Method 2 model: Method 1's gate with a size model chosen here. Person B edits sizes.py and this file.

Size options (all standard library):
  sizes="npmle"      NPMLE prior per machine type (default, = current entry); npmle_half_life
                     recency-weights the machine's own counts (B4), npmle_min_wins sets atom threshold (B5)
  sizes="dirichlet"  Dirichlet shrinkage toward type x signal average (size_strength)
  sizes="minka"      B6 control: Dirichlet-multinomial per type, alpha by Minka fixed point
  sizes="lag"        LagSizes on top of NPMLE: size counts conditioned on win/loss at size_lags,
                     shrunk toward the NPMLE prediction with lag_strength; optional size_half_life.
The gate (win probability) is untouched, so size changes are measured on their own.
"""
from method1_gate.model import Model as GateModel
from method2_sizes.sizes import NPMLESizes, DirichletSizes, LagSizes, MinkaSizes


class Model(GateModel):
    def __init__(self, sizes="npmle", size_lags=(1,), lag_strength=20.0, size_half_life=None,
                 npmle_min_wins=30, npmle_half_life=None, **kw):
        base_kind = {"dirichlet": "dirichlet", "minka": "minka"}.get(sizes, "npmle")
        super().__init__(sizes="dirichlet" if base_kind == "dirichlet" else "npmle", **kw)
        if base_kind == "minka":
            self.sizes = MinkaSizes()
        elif base_kind == "npmle":
            self.sizes = NPMLESizes(min_wins=npmle_min_wins, half_life=npmle_half_life)
        if sizes == "lag":
            self.sizes = LagSizes(self.sizes, lags=size_lags, strength=lag_strength, half_life=size_half_life)
        elif sizes not in ("npmle", "dirichlet", "minka"):
            raise ValueError(f"unknown sizes option {sizes!r}")
