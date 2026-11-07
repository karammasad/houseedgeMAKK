"""Gate+sizes only (no GBM): priors learned on train vs train+val+test histories. Tuning half."""
import sys, time
from common.evaluate import points, folds
from method4_ml.model import GATE
from method4_ml.pooled import PooledAdi
PUB = "data/main"; tune, _ = folds(PUB)
for name, kw in [("pooled all, cap 400/type", dict(max_per_type=400)), ("pooled all, no cap", dict())]:
    t0 = time.time(); m = PooledAdi(**kw, **GATE).fit(PUB); fs = time.time() - t0
    tp, r = points(PUB, m.predict(PUB, "validation"), tune)
    print(f"{name}: tuning {tp:.3f} fit {fs:.0f}s p_change_short={m.p_change_short}", {c: round(float(v), 4) for c, v in r.items()}, flush=True)
