"""Method 4 v2: gradient-boosted stacker on Method 1 + two-state size HMM. Owner: Person D.

Win probability: HistGradientBoostingClassifier on features2 (Method 1 posterior summaries for this machine,
episode change signals from the other machines, lag/rate features). Training examples are observed history
ticks t >= t_min, K random ticks per episode, from the train split and (splits="all") the HISTORIES of
validation and test too (allowed by AGENTS.md: histories only, never labels/targets/conditions).
Every feature for tick t uses ticks < t only, for every machine. Then
    P(win) = blend * GBM + (1 - blend) * Method 1.
Sizes given a win: SizeHMM (size_hmm.py), NPMLE (method2_sizes), or their average (sizes="mix").

Settings: K, t_min, splits, lr, max_leaf_nodes, min_samples_leaf, l2, max_iter, seeds, blend, sizes, workers.
variant="v1" runs the old baseline (baseline.py). Needs scikit-learn.
"""
import hashlib, os, pickle, random
from collections import defaultdict
from multiprocessing import get_context
from common.data import load_split, wins
from method1_gate.model import Model as M1
from method2_sizes.sizes import NPMLESizes
from method4_ml.features2 import episode_examples
from method4_ml.size_hmm import SizeHMM

FEATURE_VERSION = "f2.1"
HERE = os.path.dirname(os.path.abspath(__file__))
_G = {}


def _init(ST, sprior, m1cfg):
    _G.update(ST=ST, sprior=sprior, m1cfg=m1cfg)


def _episode_job(args):
    machines, seq, ticks = args
    return episode_examples(machines, seq, ticks, _G["ST"], _G["sprior"], _G["m1cfg"])


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class Model:
    def __init__(self, variant="v2", K=24, K_short=24, t_min=100, splits="all", lr=0.05, max_leaf_nodes=15,
                 min_samples_leaf=200, l2=1.0, max_iter=400, seeds=3, blend=0.8, sizes="hmm",
                 size_h=None, size_fit="train", workers=None, cache=True, **v1kw):
        self.variant = variant
        if variant == "v1":
            from method4_ml.baseline import Model as V1
            self.v1 = V1(**v1kw); return
        self.K, self.K_short, self.t_min, self.splits = K, K_short, t_min, splits
        self.gbm = dict(learning_rate=lr, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2, max_iter=max_iter, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=30)
        self.seeds, self.blend, self.sizes_kind, self.size_h, self.size_fit = seeds, blend, sizes, size_h, size_fit
        self.workers = workers or min(os.cpu_count() or 1, 8); self.cache = cache

    # ---------- feature building (cached) ----------
    def _features(self, pub, split, queries=False):
        key = (os.path.abspath(pub), split, queries, self.K, self.K_short, self.t_min, FEATURE_VERSION)
        path = os.path.join(HERE, "cache", hashlib.sha256(repr(key).encode()).hexdigest()[:16] + ".pkl")
        if self.cache and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        q, seq = load_split(pub, split); eps = defaultdict(list)
        for r in q:
            eps[r["episode_id"]].append(r)
        jobs = []
        for e in sorted(eps):
            ms = eps[e]; cut = ms[0]["cutoff"]; sub = {r["machine_id"]: seq[r["machine_id"]] for r in ms}
            if queries:
                ticks = [cut]
            else:
                rng = random.Random(_seed(e, self.K, self.t_min)); cand = list(range(self.t_min, cut))
                ticks = sorted(rng.sample(cand, min(self.K, len(cand))))
            jobs.append((ms, sub, ticks))
            # short-history augmentation: mask the past like the short_history condition (40 recent + 24 older)
            if not queries and self.K_short and all(min(v) >= 0 for v in sub.values()):
                rng = random.Random(_seed(e, "short", self.K_short, self.t_min))
                for t in sorted(rng.sample(range(max(self.t_min, 64), cut), min(self.K_short, cut - max(self.t_min, 64)))):
                    msub = {}
                    for mid, v in sub.items():
                        keep = set(range(t - 40, t + 1)) | set(rng.sample(range(t - 40), 24))
                        msub[mid] = [x if i in keep else -1 for i, x in enumerate(v[:t + 1])]
                    jobs.append((ms, msub, [t]))
        with get_context("fork").Pool(self.workers, _init, (self.m1.ST, self.m1.sprior, self.m1.cfg)) as pool:
            res = pool.map(_episode_job, jobs, chunksize=1)
        X, y, ids = [], [], []
        for (ms, sub, ticks), feats in zip(jobs, res):
            for r in ms:
                w = wins(sub[r["machine_id"]])
                for t in ticks:
                    if queries:
                        X.append(feats[(r["machine_id"], t)]); ids.append(r["query_id"])
                    elif w[t] >= 0:
                        X.append(feats[(r["machine_id"], t)]); y.append(w[t])
        out = (X, y, ids)
        if self.cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(out, f)
        return out

    # ---------- fit / predict ----------
    def fit(self, pub):
        if self.variant == "v1":
            self.v1.fit(pub); return self
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m1 = M1().fit(pub)
        q, seq = load_split(pub, "train")
        if self.size_fit == "all":  # size EM on every split's HISTORIES (no labels/targets)
            q, seq = list(q), dict(seq)
            for s in ("validation", "test"):
                if os.path.exists(os.path.join(pub, s, "history.csv")):
                    a, b = load_split(pub, s); q += a; seq.update(b)
        self.size_models = {}
        if self.sizes_kind in ("hmm", "mix"):
            self.size_models["hmm"] = SizeHMM(h=self.size_h).fit(q, seq)
        if self.sizes_kind in ("npmle", "mix"):
            self.size_models["npmle"] = NPMLESizes().fit(q, seq)
        splits = ["train", "validation", "test"] if self.splits == "all" else ["train"]
        X, y = [], []
        for s in splits:
            if os.path.exists(os.path.join(pub, s, "history.csv")):
                a, b, _ = self._features(pub, s); X += a; y += b
        self.n_train = len(X)
        self.clfs = [HistGradientBoostingClassifier(random_state=k, **self.gbm).fit(X, y) for k in range(self.seeds)]
        return self

    def predict(self, pub, split):
        if self.variant == "v1":
            return self.v1.predict(pub, split)
        q, seq = load_split(pub, split); X, _, ids = self._features(pub, split, queries=True)
        P = [0.0] * len(X)
        for c in self.clfs:
            for i, p in enumerate(c.predict_proba(X)[:, 1]):
                P[i] += p / len(self.clfs)
        pw = {qid: self.blend * p + (1 - self.blend) * x[0] for qid, p, x in zip(ids, P, X)}
        out = {}
        for r in q:
            s = seq[r["machine_id"]]; sz = [self.size_models[k].predict(r, s) for k in self.size_models]
            sz = [sum(v[i] for v in sz) / len(sz) for i in range(4)]; p = min(max(pw[r["query_id"]], 1e-6), 1 - 1e-6)
            out[r["query_id"]] = [1 - p] + [p * x for x in sz]
        return out
