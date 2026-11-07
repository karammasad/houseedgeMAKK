"""Method 4 v3: gradient-boosted stacker on top of the team's best gate + size models. Owner: Person D.

Win probability: HistGradientBoostingClassifier on features3 (summaries of adi's change-point gate for this
machine, the other episode machines' flip signals, lag/rate features). Training examples: K random observed
history ticks per episode from train plus the HISTORIES of validation/test (allowed: histories only), plus
K_short copies per long episode with the past masked like the short_history condition (40 recent + 24 older
ticks). Every feature for tick t uses ticks < t only. P(win) = blend * GBM + (1 - blend) * gate.
Sizes given a win: method2_sizes.GateSizes (couple + tied) via the adi model, or "hmm" (size_hmm.py),
or the average of both ("mix"). Everything is learned in fit() from pub/train (+ histories); nothing hardcoded.

Settings: K, K_short, t_min, splits, lr, max_leaf_nodes, min_samples_leaf, l2, max_iter, seeds, blend, sizes,
gate (kwargs for adi.model.Model), workers, cache. variant="v1" runs the old baseline (baseline.py).
Needs scikit-learn.
"""
import hashlib, json, os, pickle, random
from collections import defaultdict
from multiprocessing import get_context
from common.data import load_split, wins
from adi.model import Model as AdiModel
from method2_sizes.sizes import GateSizes
from method4_ml.pooled import pooled_split
from method4_ml.features3 import episode_examples, CELL_N
from method4_ml.size_hmm import SizeHMM

FEATURE_VERSION = "f3.2"
HERE = os.path.dirname(os.path.abspath(__file__))
# adi's win-gate configuration (adi/FINDINGS.md): wide pruned family, learned cell priors. Its built-in sizes are only a
# cheap placeholder here: the real size model is SIZES below, fit separately (optionally on every split's histories).
GATE = {"max_lag": 12, "and_pairs": True, "prior_flips": True, "prune": 0.01, "floor": 0.05, "cell_prior": "learn",
        "cell_group": "type", "p_change_short": "learn", "sizes": "npmle"}
# method2_sizes.GateSizes, Momin's FINAL (method2_sizes/round2.py): coupling to the win gate, two-state ("tied") size
# gate with a learned tie weight, joint empirical-Bayes EM.
SIZES = dict(max_lag=8, strength=4.0, p_none=0.3, learn_prior=True, cell_prior="npmle", couple=True, couple_alpha=5.0,
             couple_mix=0.1, couple_iters=20, tied=True, tie_weight="learn", joint_em=True, joint_iters=30)
_G = {}


def _init(gm):
    _G["gm"] = gm


def _episode_job(args):
    machines, seq, ticks = args
    return episode_examples(machines, seq, ticks, _G["gm"])


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _cache_path(*key):
    return os.path.join(HERE, "cache", hashlib.sha256(repr(key).encode()).hexdigest()[:16] + ".pkl")


def _load(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class GateInit:
    """Start boosting from the gate's own logit (feature column 0 = gate P(win)); trees learn only corrections."""
    def fit(self, X, y, sample_weight=None):
        import numpy as np
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        import numpy as np
        p = np.clip(np.asarray(X)[:, 0], 1e-4, 1 - 1e-4); return np.column_stack([1 - p, p])


class Model:
    def __init__(self, variant="v3", stack="hgb", n_estimators=150, max_depth=3, subsample=0.8, cells=False, K=24, K_short=24, t_min=100, splits="all", lr=0.03, max_leaf_nodes=15,
                 min_samples_leaf=1000, l2=1.0, max_iter=400, seeds=3, blend=0.4, sizes="gate", gate=None, size_kw=None, size_pool="train",
                 workers=None, cache=True, **v1kw):
        self.variant = variant
        if variant == "v1":
            from method4_ml.baseline import Model as V1
            self.v1 = V1(**v1kw); return
        self.K, self.K_short, self.t_min, self.splits = K, K_short, t_min, splits
        self.gbm = dict(learning_rate=lr, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2, max_iter=max_iter, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=30)
        self.seeds, self.blend, self.sizes_kind = seeds, blend, sizes
        self.stack, self.cells = stack, cells
        self.og = dict(learning_rate=lr, n_estimators=n_estimators, max_depth=max_depth, subsample=subsample,
                       min_samples_leaf=min_samples_leaf, max_features=0.7)
        self.gate_kw = dict(GATE if gate is None else gate)
        self.size_kw = dict(SIZES if size_kw is None else size_kw); self.size_pool = tuple(size_pool.split("+")) if isinstance(size_pool, str) else tuple(size_pool)
        self.workers = workers or min(os.cpu_count() or 1, 8); self.cache = cache

    # ---------- gate model (fit on train, cached) ----------
    def _fit_gate(self, pub):
        path = _cache_path("gate", os.path.abspath(pub), json.dumps(self.gate_kw, sort_keys=True), FEATURE_VERSION)
        gm = _load(path) if self.cache else None
        if gm is None:
            gm = AdiModel(**self.gate_kw).fit(pub)
            if self.cache:
                _save(path, gm)
        return gm

    def _fit_sizes(self, pub, gm):
        path = _cache_path("sizes", os.path.abspath(pub), json.dumps(self.gate_kw, sort_keys=True),
                           json.dumps(self.size_kw, sort_keys=True), self.size_pool, FEATURE_VERSION)
        g = _load(path) if self.cache else None
        if g is None:
            q, seq = pooled_split(pub, self.size_pool)     # histories only (train, optionally + validation/test)
            g = GateSizes(win_cfg=gm.cfg, **self.size_kw).fit(q, seq)
            if self.cache:
                _save(path, g)
        return g

    # ---------- features (cached) ----------
    def _features(self, pub, split, queries=False):
        path = _cache_path(os.path.abspath(pub), split, queries, self.K, self.K_short, self.t_min,
                           json.dumps(self.gate_kw, sort_keys=True), FEATURE_VERSION)
        hit = _load(path) if self.cache else None
        if hit is not None:
            return hit
        q, seq = load_split(pub, split); eps = defaultdict(list)
        for r in q:
            eps[r["episode_id"]].append(r)
        jobs = []
        for e in sorted(eps):
            ms = eps[e]; cut = ms[0]["cutoff"]; sub = {r["machine_id"]: seq[r["machine_id"]] for r in ms}
            if queries:
                jobs.append((ms, sub, [cut])); continue
            rng = random.Random(_seed(e, self.K, self.t_min)); cand = list(range(self.t_min, cut))
            jobs.append((ms, sub, sorted(rng.sample(cand, min(self.K, len(cand))))))
            # short-history augmentation: mask the past like the short_history condition (40 recent + 24 older)
            if self.K_short and all(min(v) >= 0 for v in sub.values()):
                rng = random.Random(_seed(e, "short", self.K_short, self.t_min)); lo = max(self.t_min, 64)
                for t in sorted(rng.sample(range(lo, cut), min(self.K_short, cut - lo))):
                    msub = {}
                    for mid, v in sub.items():
                        keep = set(range(t - 40, t + 1)) | set(rng.sample(range(t - 40), 24))
                        msub[mid] = [x if i in keep else -1 for i, x in enumerate(v[:t + 1])]
                    jobs.append((ms, msub, [t]))
        with get_context("fork").Pool(self.workers, _init, (self.gm,)) as pool:
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
            _save(path, out)
        return out

    # ---------- fit / predict ----------
    def fit(self, pub):
        if self.variant == "v1":
            self.v1.fit(pub); return self
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.gm = self._fit_gate(pub)
        self.size_models = {}
        if self.sizes_kind in ("gate", "mix"):
            self.size_models["gate"] = self._fit_sizes(pub, self.gm)
        if self.sizes_kind in ("hmm", "mix"):
            q, seq = load_split(pub, "train"); self.size_models["hmm"] = SizeHMM().fit(q, seq)
        splits = ["train", "validation", "test"] if self.splits == "all" else ["train"]
        X, y = [], []
        for s in splits:
            if os.path.exists(os.path.join(pub, s, "history.csv")):
                a, b, _ = self._features(pub, s); X += a; y += b
        self.n_train = len(X)
        X = self._sel(X)
        if self.stack == "offset":
            from sklearn.ensemble import GradientBoostingClassifier
            self.clfs = [GradientBoostingClassifier(init=GateInit(), random_state=k, **self.og).fit(X, y) for k in range(self.seeds)]
        else:
            self.clfs = [HistGradientBoostingClassifier(random_state=k, **self.gbm).fit(X, y) for k in range(self.seeds)]
        return self

    def _sel(self, X):
        """Feature matrix; cells=False drops the per-structure cell-statistics columns (7 .. 7 + 2*CELL_N)."""
        import numpy as np
        X = np.asarray(X, dtype=float)
        return X if self.cells else np.delete(X, np.s_[7:7 + 2 * CELL_N], axis=1)

    def predict(self, pub, split):
        if self.variant == "v1":
            return self.v1.predict(pub, split)
        q, seq = load_split(pub, split); X, _, ids = self._features(pub, split, queries=True)
        Xraw = X; X = self._sel(X); P = [0.0] * len(X)
        for c in self.clfs:
            for i, p in enumerate(c.predict_proba(X)[:, 1]):
                P[i] += p / len(self.clfs)
        pw = {qid: self.blend * p + (1 - self.blend) * x[0] for qid, p, x in zip(ids, P, Xraw)}
        key = (os.path.abspath(pub), split, tuple(self.size_models))   # sizes do not depend on blend / GBM: reuse
        cache = self.__dict__.setdefault("_sz", {})
        if key not in cache:
            cache[key] = {}
            for r in q:
                s = seq[r["machine_id"]]; sz = [m.predict(r, s) for m in self.size_models.values()]
                cache[key][r["query_id"]] = [sum(v[i] for v in sz) / len(sz) for i in range(4)]
        out = {}
        for r in q:
            p = min(max(pw[r["query_id"]], 1e-6), 1 - 1e-6)
            out[r["query_id"]] = [1 - p] + [p * x for x in cache[key][r["query_id"]]]
        return out
