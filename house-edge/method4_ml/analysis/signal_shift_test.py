"""Gate + sizes with signal-shifted cell priors vs plain adi priors (tuning half; sizes fixed)."""
import copy, time
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
from method4_ml.pooled import pooled_split
from method4_ml.signal_shift import attach_signal_shift
PUB = "data/main"; tune, _ = folds(PUB)
m = Model(blend=0).fit(PUB); base = m.gm; sz = m.size_models["gate"]
q, seq = load_split(PUB, "validation"); qt = [r for r in q if r["episode_id"] in tune]
SZ = {r["query_id"]: sz.predict(r, seq[r["machine_id"]]) for r in qt}
def score(gm):
    pw = gm.win_probs(qt, seq)
    return points(PUB, {r["query_id"]: [1 - pw[r["query_id"]]] + [pw[r["query_id"]] * x for x in SZ[r["query_id"]]] for r in qt}, tune)
tp, rr = score(base); print(f"plain adi priors: tuning {tp:.3f}", {c: round(float(v), 4) for c, v in rr.items()}, flush=True)
for name, pool in [("train", ("train",)), ("train+val+test histories", ("train", "validation", "test"))]:
    t0 = time.time(); gm = copy.deepcopy(base); ch = attach_signal_shift(gm, *pooled_split(PUB, pool))
    tp, rr = score(gm); print(f"signal shift ({name}, {time.time()-t0:.0f}s): tuning {tp:.3f}", {c: round(float(v), 4) for c, v in rr.items()}, flush=True)
    print("   chosen:", {t: (a_, b_, round(g, 1)) for t, (a_, b_, g) in ch.items()}, flush=True)
