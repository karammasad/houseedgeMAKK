"""Method 4 baseline: gradient boosting on leakage-safe per-tick features (D1 + D2 polarity features).
Needs scikit-learn (NOT available on the organizers' runner unless they confirm how extras install).
Settings: lr, max_leaf_nodes, min_samples_leaf, l2, max_iter, max_train (subsample of training ticks), seed."""
import random
from common.data import load_split
from method4_ml.features import machine_examples, query_features

TYPES = ("steady", "balanced", "jackpot")


def extra(q):
    return [TYPES.index(q["machine_type"]), q["signal"]]


class Model:
    def __init__(self, lr=0.1, max_leaf_nodes=31, min_samples_leaf=200, l2=1.0, max_iter=300,
                 max_train=150000, seed=0):
        self.kw = dict(learning_rate=lr, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                       l2_regularization=l2, max_iter=max_iter, early_stopping=True, random_state=seed)
        self.max_train, self.seed = max_train, seed

    def fit(self, pub):
        from sklearn.ensemble import HistGradientBoostingClassifier
        q, seq = load_split(pub, "train"); X, y = [], []
        for r in q:
            a, b, _ = machine_examples(seq[r["machine_id"]], extra(r)); X += a; y += b
        if len(X) > self.max_train:
            idx = random.Random(self.seed).sample(range(len(X)), self.max_train); X = [X[i] for i in idx]; y = [y[i] for i in idx]
        self.clf = HistGradientBoostingClassifier(**self.kw).fit(X, y)
        return self

    def predict(self, pub, split):
        q, seq = load_split(pub, split)
        X = [query_features(seq[r["machine_id"]], r["cutoff"], extra(r)) for r in q]
        P = self.clf.predict_proba(X); cls = list(self.clf.classes_); out = {}
        for r, p in zip(q, P):
            full = [float(p[cls.index(k)]) if k in cls else 1e-6 for k in range(5)]; s = sum(full)
            out[r["query_id"]] = [v / s for v in full]
        return out
