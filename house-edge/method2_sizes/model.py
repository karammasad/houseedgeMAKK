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
from method2_sizes.twolag import TwoLagSizes


class Model(GateModel):
    """win_model: None = method 1's gate (default). Or the package name of another team method whose Model
    gives the win probability (P(win) = 1 - p0 of its prediction), e.g. "adi" with win_kwargs = its settings.
    Sizes always come from this package's size model ("stacking": their gate x our size split)."""
    def fit(self, pub):
        if self.win_model:
            self.win = importlib.import_module(self.win_model + ".model").Model(**self.win_kwargs).fit(pub)
            q, seq = load_split(pub, "train"); self.sizes.fit(q, seq)
            if self.flip_prior:
                from method2_sizes.flipprior import FlipPrior
                self.flip = FlipPrior().fit(self.win, q, seq)
            if self.miss_filter:   # experiment 12: exact marginalization over unobserved gate inputs (own gate code)
                from method2_sizes.missfilter import MissFilter
                self.mf = MissFilter(self.win.signal_priors).fit(q, seq)
                if self.miss_filter == "joint":   # experiment 13: joint win x size hidden-category marginalization
                    from method2_sizes.jointfilter import JointFilter
                    g = self.sizes
                    self.jf = JointFilter(self.mf, dict(wST=g.wST, cond=g.cond, part_prior=g.part_prior,
                                                        levels=g.levels, split=g.split))
            return self
        return super().fit(pub)

    def predict(self, pub, split):
        if not self.win_model:
            return super().predict(pub, split)
        if self.joint_feedback:   # experiment 9: size -> win feedback through the joint win x size posterior
            from method2_sizes.joint import joint_parts, assemble, lag_prior_rows
            from adi.model import is_short
            rows = lag_prior_rows(self.win, self.sizes); q, seq = load_split(pub, split); out = {}
            for r in q:
                s = seq[r["machine_id"]]
                fb = self.joint_feedback == "all" or (self.joint_feedback == "short" and is_short(s))
                P = joint_parts(self.win, self.sizes, rows, r, s)
                out[r["query_id"]] = assemble(self.sizes, rows, r, s, P, fb, False, self.joint_beta)[0]
            return out
        if self.flip_prior:   # experiment 11: learned flip-time prior replaces adi's flat tau prior
            q, seq = load_split(pub, split); out = {}
            for r in q:
                s = seq[r["machine_id"]]; pw = min(1 - 1e-4, max(1e-4, self.flip.pwin(self.win, r, s)))
                out[r["query_id"]] = [1 - pw] + [pw * x for x in self.sizes.predict(r, s)]
            return out
        pw_pred = self.win.predict(pub, split)
        q, seq = load_split(pub, split); out = {}
        if self.miss_filter:
            from adi.model import is_short
        for r in q:
            p = pw_pred[r["query_id"]]; pw = 1 - p[0] / sum(p); s_ = seq[r["machine_id"]]
            if self.miss_filter and is_short(s_):   # add the information the gate discards: P(win | sampled ticks) - P(win | without)
                inc = None
                if self.miss_filter == "joint":   # P(win) increment from the joint filter (size split of FINAL is kept)
                    on, off = self.jf.run(r, s_, True), self.jf.run(r, s_, False)
                    if on is not None and off is not None:
                        inc = (1 - on[0]) - (1 - off[0])
                if inc is None:
                    inc = self.mf.pwin(r, s_, True) - self.mf.pwin(r, s_, False)
                pw = min(.999, max(.001, pw + inc))
            out[r["query_id"]] = [1 - pw] + [pw * x for x in self.sizes.predict(r, s_)]
        return out

    def __init__(self, sizes="npmle", size_lags=(1,), lag_strength=20.0, size_half_life=None,
                 npmle_min_wins=30, npmle_half_life=None,
                 gate_max_lag=8, gate_strength=4.0, gate_p_none=0.3, gate_learn_prior=True, gate_half_life=None, gate_cell_prior="beta",
                 gate_couple=False, gate_couple_alpha=5.0, gate_couple_mix=0.1, gate_couple_iters=1,
                 gate_tied=False, gate_tie_weight=0.7, gate_joint_em=False, gate_joint_iters=30,
                 gate_learn_direction=False, gate_direction_strength=20.0,
                 twolag=None, twolag_pairs="m1", twolag_eps=0.2,
                 win_model=None, win_kwargs=None, joint_feedback=None, joint_beta=1.0, flip_prior=False, miss_filter=False, **kw):
        self.miss_filter = miss_filter
        self.joint_feedback, self.joint_beta = joint_feedback, joint_beta
        self.flip_prior = flip_prior
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
                                   tied=gate_tied, tie_weight=gate_tie_weight,
                                   joint_em=gate_joint_em, joint_iters=gate_joint_iters,
                                   learn_direction=gate_learn_direction, direction_strength=gate_direction_strength)
        elif sizes not in ("npmle", "dirichlet", "minka"):
            raise ValueError(f"unknown sizes option {sizes!r}")
        if twolag:   # experiment 8: sparse two-lag expert ("alone" or Bayesian "mix" with GateSizes)
            if sizes != "gate":
                raise ValueError("twolag needs sizes='gate'")
            self.sizes = TwoLagSizes(self.sizes, mode=twolag, pairs=twolag_pairs, eps=twolag_eps)
