"""Method 3 model: Method 1 with a wider structure family and a pruned, learned prior. Person C edits this.

Settings: max_lag, max_order (3 = include XOR triples), prune (drop structures whose learned prior weight
is below this, before re-normalizing), floor (uniform mixing weight kept after pruning).
"""
from common.data import load_split
from method1_gate.gate import learn_structure_prior
from method1_gate.model import Model as GateModel


class Model(GateModel):
    def __init__(self, max_lag=8, max_order=2, prune=0.0, floor=0.1, **kw):
        super().__init__(max_lag=max_lag, max_order=max_order, **kw)
        self.prune, self.floor = prune, floor

    def fit(self, pub):
        q, seq = load_split(pub, "train")
        p = learn_structure_prior(q, seq, self.ST, self.cfg["a"], self.cfg["b"], uniform_mix=0.0)
        if self.prune > 0:
            keep = [i for i in range(len(self.ST)) if p[i] >= self.prune]
            self.ST = [self.ST[i] for i in keep]; p = [p[i] for i in keep] + [p[-1]]
        s = sum(p); p = [(1 - self.floor) * x / s + self.floor / len(p) for x in p]; s = sum(p)
        self.sprior = [x / s for x in p]
        self.sizes.fit(q, seq)
        return self
