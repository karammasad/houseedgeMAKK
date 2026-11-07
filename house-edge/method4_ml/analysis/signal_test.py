"""Gate+sizes with signal-aware cell priors vs adi per-type (no GBM). Tuning half."""
import sys, time
from common.evaluate import points, folds
from method4_ml.model import GATE
from method4_ml.signal_prior import SignalAdi
PUB = "data/main"; tune, _ = folds(PUB)
for name, kw in [("signal means, train pool", dict())]:
    t0 = time.time(); m = SignalAdi(**kw, **GATE).fit(PUB); fs = time.time() - t0
    tp, r = points(PUB, m.predict(PUB, "validation"), tune)
    print(f"{name}: tuning {tp:.3f} fit {fs:.0f}s", {c: round(float(v), 4) for c, v in r.items()}, flush=True)
    for k, v in sorted(m.cellpriors.items()): print("   ", k, [round(x, 2) for x in v], "on mean %.2f off mean %.3f" % (v[0] / (v[0] + v[1]), v[2] / (v[2] + v[3])))
