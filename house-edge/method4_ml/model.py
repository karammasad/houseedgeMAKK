"""Method 4 v5: gradient-boosted stacker on the team's best gate + size models. Owner: Person D.

Win probability: adi's change-point gate (size-split structures) with SIGNAL-SHIFTED cell priors (signal_shift.py), plus a
HistGradientBoostingClassifier on features3 (gate posterior summaries, the other episode machines' flip signals, lag/rate
features, and the size model's P(1x|win) = hidden size state). Training examples: K random observed history ticks per episode
from train plus the HISTORIES of validation/test (allowed: histories only), plus K_short copies per long episode with the past
masked like short_history (40 recent + 24 older ticks). Every feature for tick t uses ticks < t only.
P(win) = blend * GBM + (1 - blend) * gate. Sizes given a win: method2_sizes.GateSizes with Momin's FINAL settings (SIZES).
Everything is learned in fit() from pub/train (+ histories); nothing hardcoded. Needs scikit-learn.
variant="v1" runs the old baseline (baseline.py).
"""
import hashlib, json, os, pickle, random
import numpy as np
from collections import defaultdict
from multiprocessing import get_context
from common.data import load_split, wins
from adi.model import Model as AdiModel
from method2_sizes.sizes import GateSizes
from method4_ml.pooled import pooled_split
from method4_ml.signal_shift import attach_signal_shift
from method4_ml.features3 import episode_examples
from method4_ml.size_hmm import SizeHMM
from method4_ml.size_menu import fit_menu, snap
from method4_ml.logit_stack import design, fit_offset_logistic, predict as lr_predict, check_layout

FEATURE_VERSION = "f3.5"   # (menu cache key also includes it)
HERE = os.path.dirname(os.path.abspath(__file__))
# adi's win-gate configuration (adi/FINDINGS.md): wide pruned family, learned cell priors. Its built-in sizes are only a
# cheap placeholder here: the real size model is SIZES below, fit separately (optionally on every split's histories).
GATE = {"max_lag": 12, "and_pairs": True, "prior_flips": True, "prune": 0.01, "floor": 0.05, "cell_prior": "learn",
        "cell_group": "type", "p_change_short": "learn", "sizes": "npmle", "size_split": True}
# method2_sizes.GateSizes, Momin's FINAL (method2_sizes/round2.py): coupling to the win gate, two-state ("tied") size
# gate with a learned tie weight, joint empirical-Bayes EM.
SIZES = dict(max_lag=8, strength=4.0, p_none=0.3, learn_prior=True, cell_prior="npmle", couple=True, couple_alpha=5.0,
             couple_mix=0.1, couple_iters=20, tied=True, tie_weight="learn", joint_em=True, joint_iters=30)
_G = {}


def _init(gm, sz):
    _G["gm"], _G["sz"] = gm, sz


def _episode_job(args):
    machines, seq, ticks = args
    return episode_examples(machines, seq, ticks, _G["gm"], _G["sz"])


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
    def __init__(self, variant="v3", stack="hgb", n_estimators=150, max_depth=3, subsample=0.8, K=6, K_short=6, t_min=100, splits="all", lr=0.03, max_leaf_nodes=15,
                 min_samples_leaf=1000, l2=1.0, max_iter=400, seeds=3, blend=0.4, sizes="gate", gate=None, size_kw=None, size_pool="train", signal_shift=True, size_feat=True, size_stack=False, size_blend=0.5, tick_seed=0, bags=5, lr_weight=0.5, lr_spec=("d", "chg"), lr_l2=1.0, cshift=0.0, cs_clip=10.0, size_menu=True, menu_k=32, menu_check=True,
                 workers=None, cache=True, **v1kw):
        self.variant = variant
        if variant == "v1":
            from method4_ml.baseline import Model as V1
            self.v1 = V1(**v1kw); return
        self.K, self.K_short, self.t_min, self.splits = K, K_short, t_min, splits
        self.gbm = dict(learning_rate=lr, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2, max_iter=max_iter, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=30)
        self.seeds, self.blend, self.sizes_kind, self.signal_shift = seeds, blend, sizes, signal_shift
        self.size_feat, self.size_stack, self.size_blend = size_feat, size_stack, size_blend
        self.tick_seed = tick_seed   # which random history ticks become training examples (0 = original sample)
        self.lr_weight, self.lr_spec, self.lr_l2 = lr_weight, tuple(lr_spec), lr_l2
        self.size_menu, self.menu_k, self.menu_check = size_menu, menu_k, menu_check   # menu_check: keep a type's snap only if the no-target self-check supports it   # snap sizes onto a learned 2-vector menu per type (size_menu.py)
        self.cshift, self.cs_clip = cshift, cs_clip   # covariate-shift correction strength (0 = off): weights ~ (q(x)/p(x))**cshift
        self.bags = bags             # bagging: stackers trained on `bags` independent tick samples, probabilities averaged
        self.stack = stack
        self.og = dict(learning_rate=lr, n_estimators=n_estimators, max_depth=max_depth, subsample=subsample,
                       min_samples_leaf=min_samples_leaf, max_features=0.7)
        self.gate_kw = dict(GATE if gate is None else gate)
        self.size_kw = dict(SIZES if size_kw is None else size_kw); self.size_pool = tuple(size_pool.split("+")) if isinstance(size_pool, str) else tuple(size_pool)
        self.workers = workers or min(os.cpu_count() or 1, 8); self.cache = cache

    # ---------- gate model (fit on train, cached) ----------
    def _fit_gate(self, pub):
        path = _cache_path("gate", os.path.abspath(pub), json.dumps(self.gate_kw, sort_keys=True), self.signal_shift, FEATURE_VERSION)
        gm = _load(path) if self.cache else None
        if gm is None:
            gm = AdiModel(**self.gate_kw).fit(pub)
            if self.signal_shift:      # signal-dependent cell priors, learned by marginal likelihood on train histories
                attach_signal_shift(gm, *pooled_split(pub, ("train",)))
            if self.cache:
                _save(path, gm)
        return gm

    def _fit_sizes(self, pub, gm):
        path = _cache_path("sizes", os.path.abspath(pub), json.dumps(self.gate_kw, sort_keys=True), self.signal_shift,
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
        path = _cache_path(os.path.abspath(pub), split, queries, self.K, self.K_short, self.t_min, self.tick_seed if not queries else 0,
                           json.dumps(self.gate_kw, sort_keys=True), self.signal_shift, self.size_feat,
                           json.dumps(self.size_kw, sort_keys=True), FEATURE_VERSION)
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
            rng = random.Random(_seed(e, self.K, self.t_min, *([self.tick_seed] if self.tick_seed else []))); cand = list(range(self.t_min, cut))
            jobs.append((ms, sub, sorted(rng.sample(cand, min(self.K, len(cand))))))
            # short-history augmentation: mask the past like the short_history condition (40 recent + 24 older)
            if self.K_short and all(min(v) >= 0 for v in sub.values()):
                rng = random.Random(_seed(e, "short", self.K_short, self.t_min, *([self.tick_seed] if self.tick_seed else []))); lo = max(self.t_min, 64)
                for t in sorted(rng.sample(range(lo, cut), min(self.K_short, cut - lo))):
                    msub = {}
                    for mid, v in sub.items():
                        keep = set(range(t - 40, t + 1)) | set(rng.sample(range(t - 40), 24))
                        msub[mid] = [x if i in keep else -1 for i, x in enumerate(v[:t + 1])]
                    jobs.append((ms, msub, [t]))
        init_args = (self.gm, self.size_models.get("gate") if self.size_feat else None)
        res = None
        if self.workers > 1:
            try:      # parallel feature building; falls back to serial (identical results) if the runner forbids Pool/shared memory
                with get_context("fork").Pool(self.workers, _init, init_args) as pool:
                    res = pool.map(_episode_job, jobs, chunksize=1)
            except (OSError, PermissionError, ImportError, ValueError, NotImplementedError) as e:
                print(f"method4_ml: multiprocessing unavailable ({type(e).__name__}: {e}); building features serially", flush=True)
                res = None
        if res is None:
            _init(*init_args); res = [_episode_job(j) for j in jobs]
        X, y, ids = [], [], []
        for (ms, sub, ticks), feats in zip(jobs, res):
            for r in ms:
                w = wins(sub[r["machine_id"]])
                for t in ticks:
                    if queries:
                        X.append(feats[(r["machine_id"], t)]); ids.append(r["query_id"])
                    elif w[t] >= 0:
                        X.append(feats[(r["machine_id"], t)]); y.append(sub[r["machine_id"]][t])   # payout (0/1/2/5/20)
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
        self.menu = None
        if self.size_menu and "gate" in self.size_models:     # exact two-state size menu, learned from HISTORY ticks only
            path = _cache_path("menu", os.path.abspath(pub), json.dumps(self.gate_kw, sort_keys=True), self.signal_shift,
                               json.dumps(self.size_kw, sort_keys=True), self.size_pool, tuple(splits), self.menu_k, self.menu_check, FEATURE_VERSION)
            hit = _load(path) if self.cache else None
            if hit is None:
                rep = {}; menu = fit_menu(pub, splits, self.size_models["gate"], self.menu_k, self.workers, check=self.menu_check, report=rep)
                hit = (menu, rep)
                if self.cache:
                    _save(path, hit)
            self.menu, self.menu_report = hit
            if self.menu_check:
                print("method4_ml size-menu self-check:", {t: (round(v["t"], 1), "keep" if v["kept"] else "OFF") for t, v in self.menu_report.items()}, flush=True)
        import numpy as np
        self.clfs = []; base_seed = self.tick_seed; Xall, yall, Wall = [], [], []
        Xq = None
        if self.cshift > 0:      # query-regime features of every split (features only, no labels): the target of the shift correction
            Xq = np.vstack([self._sel(self._features(pub, s, queries=True)[0]) for s in splits if os.path.exists(os.path.join(pub, s, "queries.csv"))])
        for bag in range(self.bags):     # each bag: its own random training ticks (selection-luck reduction, analysis/bagging.py)
            self.tick_seed = base_seed + bag
            X, y = [], []
            for s in splits:
                if os.path.exists(os.path.join(pub, s, "history.csv")):
                    a, b, _ = self._features(pub, s); X += a; y += b
            self.n_train = len(X); ypay = y; y = [int(v > 0) for v in ypay]
            X = self._sel(X); Xall.append(X); yall += y
            sw = self._shift_weights(X, Xq) if Xq is not None else None
            Wall.append(np.ones(len(X)) if sw is None else sw)
            if self.stack == "offset":
                from sklearn.ensemble import GradientBoostingClassifier
                self.clfs += [GradientBoostingClassifier(init=GateInit(), random_state=k, **self.og).fit(X, y) for k in range(self.seeds)]
            else:
                self.clfs += [HistGradientBoostingClassifier(random_state=k, **self.gbm).fit(X, y, sample_weight=Wall[-1] if Xq is not None else None) for k in range(self.seeds)]
        self.tick_seed = base_seed
        self.beta = None
        if self.lr_weight > 0:   # structured offset-logistic correction of the gate's P(win) (adi round 7 idea, extended)
            import numpy as np
            XA = np.vstack(Xall); check_layout(XA); yA = np.asarray(yall)
            off = np.log(np.clip(XA[:, 0], 1e-4, 1 - 1e-4) / (1 - np.clip(XA[:, 0], 1e-4, 1 - 1e-4)))
            self.beta = fit_offset_logistic(design(XA, self.lr_spec), yA, off, self.lr_l2, w=np.concatenate(Wall))
        self.sclfs = []
        if self.size_stack:    # P(1x | win) stacker on the same features, winning ticks only
            wi = [i for i, v in enumerate(ypay) if v > 0]; Xs = X[wi]; ys = [int(ypay[i] == 1) for i in wi]
            self.sclfs = [HistGradientBoostingClassifier(random_state=k, **self.gbm).fit(Xs, ys) for k in range(self.seeds)]
        return self

    def _shift_weights(self, X, Xq):
        """Density-ratio weights q(x)/p(x) for training rows X vs query-regime rows Xq (discriminator on features; no labels)."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        Z = np.vstack([X, Xq]); d = np.r_[np.zeros(len(X)), np.ones(len(Xq))]
        rs = np.random.RandomState(0); idx = rs.permutation(len(Z)); half = len(Z) // 2; A, B = idx[:half], idx[half:]
        pr = np.zeros(len(Z))      # 2-fold cross-fitted so the discriminator does not memorise its own rows
        for tr, te in ((A, B), (B, A)):
            c = HistGradientBoostingClassifier(max_leaf_nodes=8, learning_rate=0.05, max_iter=150, min_samples_leaf=50, l2_regularization=5.0, random_state=0).fit(Z[tr], d[tr])
            pr[te] = c.predict_proba(Z[te])[:, 1]
        p = np.clip(pr[:len(X)], 1e-4, 1 - 1e-4); w = (p / (1 - p)) * (len(X) / len(Xq)) ** 0   # odds ~ density ratio (up to a constant)
        w = np.clip(w / w.mean(), 1 / self.cs_clip, self.cs_clip) ** self.cshift
        return w / w.mean()

    @staticmethod
    def _sel(X):
        import numpy as np
        return np.asarray(X, dtype=float)

    def predict(self, pub, split):
        if self.variant == "v1":
            return self.v1.predict(pub, split)
        q, seq = load_split(pub, split); X, _, ids = self._features(pub, split, queries=True)
        Xraw = X; X = self._sel(X); P = [0.0] * len(X)
        for c in self.clfs:
            for i, p in enumerate(c.predict_proba(X)[:, 1]):
                P[i] += p / len(self.clfs)
        pw = {qid: self.blend * p + (1 - self.blend) * x[0] for qid, p, x in zip(ids, P, Xraw)}
        if self.beta is not None and self.lr_weight > 0:
            import numpy as np
            XQ = np.asarray(Xraw, float); check_layout(XQ)
            off = np.log(np.clip(XQ[:, 0], 1e-4, 1 - 1e-4) / (1 - np.clip(XQ[:, 0], 1e-4, 1 - 1e-4)))
            lr = lr_predict(design(XQ, self.lr_spec), self.beta, off)
            pw = {qid: self.lr_weight * float(l) + (1 - self.lr_weight) * pw[qid] for qid, l in zip(ids, lr)}
        key = (os.path.abspath(pub), split, tuple(self.size_models))   # sizes do not depend on blend / GBM: reuse
        cache = self.__dict__.setdefault("_sz", {})
        if key not in cache:
            cache[key] = {}
            for r in q:
                s = seq[r["machine_id"]]; sz = [m.predict(r, s) for m in self.size_models.values()]
                cache[key][r["query_id"]] = [sum(v[i] for v in sz) / len(sz) for i in range(4)]
        P1 = None
        if self.sclfs and self.size_blend > 0:
            P1 = [0.0] * len(X)
            for c in self.sclfs:
                for i, p in enumerate(c.predict_proba(X)[:, 1]):
                    P1[i] += p / len(self.sclfs)
            P1 = dict(zip(ids, P1))
        out = {}
        for r in q:
            qid = r["query_id"]; p = min(max(pw[qid], 1e-6), 1 - 1e-6); sz = cache[key][qid]
            if P1 is not None:      # blend the size stacker into P(1x | win); rescale the big-win split to sum to 1
                a = (1 - self.size_blend) * sz[0] + self.size_blend * P1[qid]; a = min(max(a, 1e-4), 1 - 1e-4)
                rest = sum(sz[1:]); sz = [a] + [x * (1 - a) / rest for x in sz[1:]] if rest > 0 else sz
            if self.menu is not None:
                sz = snap(sz, self.menu.get(r["machine_type"]))
            out[qid] = [1 - p] + [p * x for x in sz]
        return out
