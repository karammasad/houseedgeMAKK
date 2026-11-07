"""Method 1 win prob x SizeHMM sizes, tuning half.  python -m method4_ml.analysis.test_sizes [h]"""
import pickle, sys, time
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.size_hmm import SizeHMM
PUB = "data/main"
pred = pickle.load(open("method4_ml/analysis/m1_val.pkl", "rb"))
tq, ts = load_split(PUB, "train"); vq, vs = load_split(PUB, "validation")
tune, _ = folds(PUB)
for h in ([None] if len(sys.argv) < 2 else [None if a == "None" else float(a) for a in sys.argv[1:]]):
    t0 = time.time(); m = SizeHMM(h=h).fit(tq, ts)
    print(f"h={h} learned h={m.h:.5f} pi={m.pi0:.3f} fit {time.time()-t0:.1f}s")
    for t, E in m.E.items(): print("  ", t, [[round(x, 3) for x in e] for e in E])
    out = {}
    for r in vq:
        pw = 1 - pred[r["query_id"]][0]; out[r["query_id"]] = [1 - pw] + [pw * x for x in m.predict(r, vs[r["machine_id"]])]
    tp, rr = points(PUB, out, tune); print(f"  tuning {tp:.3f}", {c: round(v, 4) for c, v in rr.items()})
