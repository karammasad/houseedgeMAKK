"""Standard-library version of method4_ml v5 (Kareem's stacker), using exported trees. Owner: Adi.

method4_ml v5 = adi gate (size split, signal-shifted priors) + method 2 GateSizes + a gradient-boosted stacker on
leakage-safe features. Everything except the trees is standard library. This model reuses method4_ml's own
gate / size / feature code (imported read-only, unchanged) and replaces the scikit-learn trees with the same trees
exported to JSON by adi/stack_export.py, evaluated in pure Python. No scikit-learn needed at prediction time.

  python -m common.evaluate --method adi.stack --data data/main

Settings: trees = path of the exported JSON (default adi/stack/trees_main.json). All method4 settings come from the
JSON (the ones the trees were trained with), so features always match the trees.
"""
import json, math, os
from common.data import load_split
from method4_ml.model import Model as M4

HERE = os.path.dirname(os.path.abspath(__file__))


def tree_raw(nodes, x):
    i = 0
    while True:
        leaf, f, thr, miss_left, left, right, val = nodes[i]
        if leaf:
            return val
        v = x[f]
        if v != v:  # NaN
            i = left if miss_left else right
        else:
            i = left if v <= thr else right


class Model(M4):
    def __init__(self, trees="trees_main.json", workers=None):
        path = trees if os.path.isabs(trees) else os.path.join(HERE, trees)
        with open(path) as f:
            self.export = json.load(f)
        kw = dict(self.export["m4_kwargs"]); kw["cache"] = False
        if workers is not None:
            kw["workers"] = workers
        super().__init__(**kw)

    def fit(self, pub):
        """Fits only the standard-library parts (gate with signal shift, GateSizes); the trees are loaded."""
        self.gm = self._fit_gate(pub)
        self.size_models = {"gate": self._fit_sizes(pub, self.gm)}
        return self

    def win_stack(self, X):
        out = []
        for x in X:
            p = 0.0
            for mdl in self.export["models"]:
                raw = mdl["baseline"] + sum(tree_raw(t, x) for t in mdl["trees"])
                p += 1 / (1 + math.exp(-raw))
            out.append(p / len(self.export["models"]))
        return out

    def predict(self, pub, split):
        q, seq = load_split(pub, split)
        X, _, ids = self._features(pub, split, queries=True)
        P = self.win_stack(X)
        pw = {qid: self.blend * p + (1 - self.blend) * x[0] for qid, p, x in zip(ids, P, X)}
        out = {}
        for r in q:
            qid = r["query_id"]; p = min(max(pw[qid], 1e-6), 1 - 1e-6)
            sz = self.size_models["gate"].predict(r, seq[r["machine_id"]])
            out[qid] = [1 - p] + [p * v for v in sz]
        return out
