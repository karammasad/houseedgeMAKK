"""Generic validation scorer for any candidate's pickled prediction dict {query_id: [p0,p1,p2,p5,p20]}.
Works for thanos (build/thanos_r2/validation.pkl), or any other method's pickled predict(pub, "validation") output.

  python3 round2/score_pred.py --data data/adaptation --pkl build/thanos_r2/validation.pkl --name thanos
"""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import public_dir
from common.evaluate import folds, points, CONDITIONS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--pkl", required=True, help="pickle of {query_id: [p0..p20]} for the validation split")
ap.add_argument("--name", default="candidate")
ap.add_argument("--out", default=None, help="optional json result path")
a = ap.parse_args()

pub = public_dir(a.data)
pred = pickle.load(open(a.pkl, "rb"))
meta = json.load(open(os.path.join(pub, "metadata.json")))
maxp = float(meta.get("max_points", 60))
tune, audit = folds(pub)

tp, tr = points(pub, pred, tune, maxp)
ap_, ar = points(pub, pred, audit, maxp)
fp, fr = points(pub, pred, None, maxp)

res = dict(name=a.name, max_points=maxp, tuning=tp, audit=ap_, full=fp,
           tuning_ratios=tr, audit_ratios=ar, full_ratios=fr)
print(f"[{a.name}] tuning {tp:.4f}  audit {ap_:.4f}  full {fp:.4f}  (max {maxp:g})")
for c in CONDITIONS:
    print(f"   {c:14s} tuning {tr[c]:.4f}  audit {ar[c]:.4f}  full {fr[c]:.4f}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
