"""Fit a method, dump validation+test predictions to JSON (analysis cache)."""
import importlib, json, sys, time
from common.data import public_dir
meth, out = sys.argv[1], sys.argv[2]
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
pub = public_dir("data/main"); t0 = time.time()
m = importlib.import_module(f"{meth}.model").Model(**kw).fit(pub)
res = {s: m.predict(pub, s) for s in ("validation", "test")}
json.dump(res, open(out, "w")); print(meth, "done", time.time() - t0, getattr(m, "kappa_fit", ""), getattr(m, "kappa_evidence", ""))
