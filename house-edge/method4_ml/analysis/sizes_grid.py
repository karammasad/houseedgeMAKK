"""Tune GateSizes hyperparameters with the win probability held fixed (tuning half only).
python -m method4_ml.analysis.sizes_grid '{"gate_strength":[2,4,8]}'  (one-at-a-time around adi's best)"""
import json, sys, time
from common.data import load_split
from common.evaluate import points, folds
from method2_sizes.sizes import GateSizes
from method4_ml.model import Model, GATE
PUB = "data/main"; tune, _ = folds(PUB)
m = Model(blend=0).fit(PUB); base = m.predict(PUB, "validation")
vq, vs = load_split(PUB, "validation"); tq, ts = load_split(PUB, "train")
DEF = dict(max_lag=8, strength=4.0, p_none=0.3, learn_prior=True, half_life=None, cell_prior="npmle", couple=True,
           couple_alpha=5.0, couple_mix=0.1, couple_iters=20, win_cfg=m.gm.cfg, tied=True, tie_weight=0.7)
def score(**over):
    t0 = time.time(); g = GateSizes(**{**DEF, **over}).fit(tq, ts); out = {}
    for r in vq:
        pw = 1 - base[r["query_id"]][0]; out[r["query_id"]] = [1 - pw] + [pw * x for x in g.predict(r, vs[r["machine_id"]])]
    tp, rr = points(PUB, out, tune); return tp, rr, time.time() - t0
grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
print("default:", "%.3f" % score()[0], flush=True)
for k, vals in grid.items():
    for v in vals:
        tp, rr, sec = score(**{k: v}); print(f"{k}={v}: tuning {tp:.3f} ({sec:.0f}s)", flush=True)
