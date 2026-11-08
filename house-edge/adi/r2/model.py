"""adi round-two model: a copy of the 58.21 main model (adi.sizewin) with every data-specific assumption widened so
it is re-learned from the new kit. Owner: Adi.

Main model (adi.sizewin) = method2_sizes FINAL (adi gate + GateSizes) + size-state logistic correction. In main,
some choices were fixed to what the main data showed. Round two "changes which temporal relationships recur", so:

  gate structures   XOR pairs AND XOR triples at lags 1..12, plus AND-family pairs; the train-learned prior is pruned
                    (prune 0.01), so only structures the new train data supports survive
  size split        cells split at BOTH lags of each kept pair (main: smaller lag only)
  size model        GateSizes size-gate lags 1..L, L = max(8, longest lag among the win structures the gate kept)
                    (adaptive: 8 on main data, which scores best there; widens if round two's rules use longer lags)
  everything else   learned in fit() from the round's train histories, as in main: cell priors per type with
                    signal slopes, short-log flip prior, structure prior, GateSizes coupling / tie weight /
                    joint EM, the size-state correction

  python -m common.evaluate --method adi.r2 --data data/adaptation
  python -m adi.r2.run --data data/adaptation --main data/main --out build/r2      (9 PM runbook)
"""
import importlib
from common.data import load_split
from adi.sizewin.model import Model as SizeWin
from method2_sizes.model import Model as M2

ADI_R2 = {"max_lag": 12, "xor_order": 3, "and_pairs": True, "prior_flips": True, "prune": 0.01, "floor": 0.05,
          "cell_prior": "learn", "cell_group": "type", "p_change_short": "learn", "size_split": "both",
          "cell_signal": True}
BASE_R2 = dict(sizes="gate", gate_cell_prior="npmle", gate_couple=True, gate_couple_iters=20, gate_tied=True,
               gate_tie_weight="learn", gate_joint_em=True, gate_max_lag="auto", win_model="adi", win_kwargs=ADI_R2)


class Model(SizeWin):
    def __init__(self, base=None, win=None, **kw):
        b = dict(base) if base is not None else dict(BASE_R2)
        if win:
            b["win_kwargs"] = dict(b["win_kwargs"], **win)
        super().__init__(base=b, **kw)

    def _make_base(self, pub):
        """method2_sizes Model with the size-lag range chosen from the fitted win gate when gate_max_lag == "auto"."""
        b = dict(self.base_kw); auto = b.get("gate_max_lag") == "auto"
        if auto:
            b["gate_max_lag"] = 8
        m = M2(**b)
        m.win = importlib.import_module(m.win_model + ".model").Model(**m.win_kwargs).fit(pub)
        if auto:
            lags = [l for f, st in m.win.ST for l in st[:2 if f in ("xs", "xst") else len(st)]]
            self.size_max_lag = max([8] + lags)
            m.sizes.max_lag = self.size_max_lag
        q, seq = load_split(pub, "train"); m.sizes.fit(q, seq)
        return m
