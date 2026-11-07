"""Gate hyperparameter grid (tuning half only): re-predict with changed gate settings, learned priors held fixed.
python -m method4_ml.analysis.gate_grid"""
import itertools, time
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
from adi import gate as G
PUB = "data/main"; tune, _ = folds(PUB)
gm = Model(blend=0).fit(PUB).gm
q, seq = load_split(PUB, "validation"); q = [r for r in q if r["episode_id"] in tune]
SZ = {r["query_id"]: gm.sizes.predict(r, seq[r["machine_id"]]) for r in q}
def score(**over):
    out = {}
    for r in q:
        s = seq[r["machine_id"]]; cfg = {**gm.cfg, **{k: v for k, v in over.items() if k in gm.cfg}}
        if sum(x < 0 for x in s) > len(s) / 2: cfg["p_change"] = over.get("p_change_short", gm.p_change_short)
        pw = G.win_probability(r, s, gm.ST, gm.sprior, cellprior=gm._cp(r), **cfg)
        out[r["query_id"]] = [1 - pw] + [pw * x for x in SZ[r["query_id"]]]
    return points(PUB, out, tune)[0]
print("baseline", round(score(), 3), flush=True)
for k, vals in dict(p_change=[.2, .35, .65, .8], tau_min=[10, 20, 30, 60, 80], step=[1, 4], a=[.25, 1.0], b=[.25, 1.0],
                    p_change_short=[.01, .02, .1, .2, .4]).items():
    print(k, {v: round(score(**{k: v}), 3) for v in vals}, flush=True)
