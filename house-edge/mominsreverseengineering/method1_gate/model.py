"""Method 1 model = gate change-point win probability x Method 2 size model. This is the current entry."""
from common.data import load_split
from method1_gate.gate import structures, learn_structure_prior, win_probability
from method2_sizes.sizes import NPMLESizes, DirichletSizes


class Model:
    def __init__(self, max_lag=8, max_order=2, a=.5, b=.5, p_change=.5, step=2, tau_min=40,
                 sizes="npmle", size_strength=30.0):
        self.cfg = dict(a=a, b=b, p_change=p_change, step=step, tau_min=tau_min)
        self.ST = structures(max_lag, max_order)
        self.sizes = NPMLESizes() if sizes == "npmle" else DirichletSizes(size_strength)

    def fit(self, pub):
        q, seq = load_split(pub, "train")
        self.sprior = learn_structure_prior(q, seq, self.ST, self.cfg["a"], self.cfg["b"])
        self.sizes.fit(q, seq)
        return self

    def predict(self, pub, split):
        q, seq = load_split(pub, split); out = {}
        for r in q:
            s = seq[r["machine_id"]]
            pw = win_probability(r, s, self.ST, self.sprior, **self.cfg)
            out[r["query_id"]] = [1 - pw] + [pw * x for x in self.sizes.predict(r, s)]
        return out
