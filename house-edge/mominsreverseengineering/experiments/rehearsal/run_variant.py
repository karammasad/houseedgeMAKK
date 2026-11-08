"""Test A runner: python3 run_variant.py <public dir> <out dir> <0|1>   (1 = apply the no-gate adaptation). Thanos code from the read-only 8e00a50 worktree, unmodified."""
import os, sys, time, json, hashlib
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[v] = "2"
WT = "/private/tmp/claude-501/-Users-mominhassan-houseedgeMAKK/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad/thanos_wt/house-edge"
sys.path.insert(0, WT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.data import public_dir, write_predictions
from thanos import egp
from thanos.model import Model
import nogate_fix
pub = public_dir(sys.argv[1]); out = sys.argv[2]; fix = sys.argv[3] == "1"; os.makedirs(out, exist_ok=True)
if fix: nogate_fix.apply(egp)
t0 = time.time(); M = Model("all", verbose=True, tickset="own").fit(pub); t1 = time.time(); P = M.predict(pub, "test"); t2 = time.time()
write_predictions(os.path.join(out, "submission.csv"), P)
H = M.H; info = dict(fix=fix, fit_s=round(t1 - t0, 1), predict_s=round(t2 - t1, 1), wall_s=round(t2 - t0, 1), K=len(H.K), nogate_mass=[round(float(__import__("math").exp(H.lpi[i])), 4) for i, (G, l) in enumerate(H.K) if len(G) == 0],
                     sha256=hashlib.sha256(open(os.path.join(out, "submission.csv"), "rb").read()).hexdigest())
json.dump(info, open(os.path.join(out, "run.json"), "w"), indent=1); print(info)
