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
import csv
from xm import xfeat, adjust          # cross-machine P(win) adjustment (history-only), beta fixed at 1.94 (train-only MLE), lags 1..8
P = M.predict(pub, "test")
qs = list(csv.DictReader(open(os.path.join(pub, "test", "queries.csv")))); hs = list(csv.DictReader(open(os.path.join(pub, "test", "history.csv"))))
F = xfeat(qs, hs, range(1, 9))
P = {q: adjust(v, F[q], 1.94) for q, v in P.items()}
P = {q: [x / sum(v) for x in v] for q, v in P.items()}
write_predictions(a.out, P)
print("wrote", a.out)
