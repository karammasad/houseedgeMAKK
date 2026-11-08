"""Self-contained entry: Thanos E2 (exact generator posterior). Fits every hyperparameter from the released HISTORIES (train + validation + test, pre-cutoff
observations only; labels, targets and conditions are never opened), then writes the test forecast.  python predict.py --data /data --out /output/submission.csv"""
import argparse, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.data import public_dir, write_predictions
from thanos.model import Model

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/data"); ap.add_argument("--out", default="/output/submission.csv"); a = ap.parse_args()
pub = public_dir(a.data)
M = Model("all", verbose=False, tickset="own").fit(pub)
write_predictions(a.out, M.predict(pub, "test"))
print("wrote", a.out)
