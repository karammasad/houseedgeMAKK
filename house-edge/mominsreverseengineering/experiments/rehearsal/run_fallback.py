"""Frozen fallback for the rehearsal: method4_ml v9 + shared-base, default settings, transductive, no selection, no caches, 2 workers + 2 BLAS threads.
  PYTHONPATH=. python3 experiments/rehearsal/run_fallback.py <public dir> <out dir>      (run from mominsreverseengineering/)"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[v] = "2"
import hashlib, json, pickle, resource, time
from common.data import public_dir, write_predictions
from method4_ml.model import Model
pub = public_dir(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True); t0 = time.time()
m = Model(shared_base=True, workers=2, cache=False).fit(pub); t1 = time.time()
P = m.predict(pub, "test"); t2 = time.time(); path = os.path.join(out, "submission.csv"); write_predictions(path, P); pickle.dump(P, open(os.path.join(out, "test.pkl"), "wb"))
k = 1024 * 1024 if sys.platform == "darwin" else 1024
json.dump(dict(fit_s=round(t1 - t0, 1), predict_s=round(t2 - t1, 1), wall_s=round(t2 - t0, 1), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / k, 1),
               peak_rss_children_mb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / k, 1), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest()), open(os.path.join(out, "run.json"), "w"), indent=1)
print(open(os.path.join(out, "run.json")).read())
