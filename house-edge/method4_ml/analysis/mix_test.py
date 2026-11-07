"""Gate+sizes with mixture-of-Betas cell priors (learned on pooled histories) vs adi's Beta priors. Tuning half."""
import copy, sys, time
from common.evaluate import points, folds
from method4_ml.model import Model
from method4_ml.pooled import pooled_split
from method4_ml.mixprior import attach_mixture
PUB = "data/main"; tune, _ = folds(PUB)
base = Model(blend=0).fit(PUB).gm
for pool, by in [(("train",), "type"), (("train",), "type_signal"), (("train", "validation", "test"), "type_signal")]:
    t0 = time.time(); gm = attach_mixture(copy.deepcopy(base), *pooled_split(PUB, pool), by=by); fs = time.time() - t0
    from common.data import load_split
    q, seq = load_split(PUB, "validation"); out = {}
    for r in q:
        s = seq[r["machine_id"]]; cfg = dict(gm.cfg)
        if gm.p_change_short is not None and sum(x < 0 for x in s) > len(s) / 2: cfg["p_change"] = gm.p_change_short
        from adi import gate as G
        pw = G.win_probability(r, s, gm.ST, gm.sprior, cellprior=gm._cp(r), **cfg)
        out[r["query_id"]] = [1 - pw] + [pw * x for x in gm.sizes.predict(r, s)]
    tp, rr = points(PUB, out, tune)
    print(f"pool={len(pool)} by={by}: tuning {tp:.3f} (attach {fs:.0f}s)", {c: round(float(v), 4) for c, v in rr.items()}, flush=True)
