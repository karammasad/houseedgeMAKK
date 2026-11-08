"""Self-contained entry: Thanos (exact generator posterior, Round 1 code path: adapt=False) + correlated on/off rate prior (within machine).
Fits every hyperparameter from the released HISTORIES (train + validation + test, pre-cutoff observations only; labels, targets and conditions
are never opened). rho (one number per machine type) is a method-of-moments estimate from TRAIN machines' own-history posteriors. Each
machine's forecast uses only its own pre-cutoff history.  python predict.py --data /data --out /output/submission.csv"""
import argparse, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.data import load_split, public_dir, write_predictions
from thanos.model import Model
from thanos import egp, episode_effect as EE
from thanos.xm_adjust import adjust

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/data"); ap.add_argument("--out", default="/output/submission.csv")
ap.add_argument("--val-out", default="", help="optional: also write validation forecasts here (verification only)"); a = ap.parse_args()
pub = public_dir(a.data)
M = Model("all", verbose=False, tickset="own", adapt=False).fit(pub); H = M.H
egp.TICKSET = "own"; egp.WINDOW_FRAC = getattr(H, "window_frac", .45)


def machines(sp):
    ms = egp.machines_from_split(*load_split(pub, sp)); egp.set_size_def(ms, getattr(H, "size_def", "big")); return ms


tr = machines("train"); RHO = EE.rho_hat(tr, H, EE.stage1(tr, H)); print("rho", RHO, flush=True)
for sp, path in (("test", a.out), ("validation", a.val_out)):
    if path:
        ms = machines(sp); P = EE.corr_predict(ms, H, EE.stage1(ms, H), RHO)
        q, s = load_split(pub, sp); write_predictions(path, adjust(q, s, P, beta=1.94)); print("wrote", path, flush=True)
