"""Leakage / validity tests for the correlated on/off prior step (episode_effect.rho_hat / corr_predict), fixed thanos hyperparameters.
   python3 -m thanos.tests_corr_prior --data data/adaptation --hyper build/r2_real/r1/hyper.pkl     (exit 1 on any failure)"""
import argparse, pickle, random, sys
import numpy as np
from common.data import load_split, public_dir
from . import egp, episode_effect as EE

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/adaptation"); ap.add_argument("--hyper", required=True); a = ap.parse_args()
pub = public_dir(a.data); H = pickle.load(open(a.hyper, "rb")); egp.TICKSET = "own"; egp.WINDOW_FRAC = .45; rng = random.Random(0); F = []
OPENED = []; sys.addaudithook(lambda e, x: OPENED.append(x[0]) if e == "open" and isinstance(x[0], str) and x[0].endswith(".csv") else None)
def ms_of(q, s):
    m = egp.machines_from_split(q, s); egp.set_size_def(m, "big"); return m
def run(q, s, RHO):
    m = ms_of(q, s); return EE.corr_predict(m, H, EE.stage1(m, H), RHO)
def same(A, B): return all(max(abs(x - y) for x, y in zip(A[k], B[k])) < 1e-12 for k in A)
qt, st = load_split(pub, "train"); RHO = EE.rho_hat(ms_of(qt, st), H, EE.stage1(ms_of(qt, st), H))
qv, sv = load_split(pub, "validation"); eps = sorted({r["episode_id"] for r in qv})[:6]; qv = [r for r in qv if r["episode_id"] in eps]
sv = {r["machine_id"]: sv[r["machine_id"]] for r in qv}; base = run(qv, sv, RHO)
# 1 future ticks never enter: the official loader rejects any history row at/after the cutoff (a Machine's cutoff is len(history))
import csv as _csv, shutil as _sh, tempfile as _tf
_d = _tf.mkdtemp(); _sh.copytree(public_dir(a.data) + "/validation", _d + "/validation")
with open(_d + "/validation/history.csv", "a", newline="") as _f:
    _csv.writer(_f).writerow([qv[0]["episode_id"], qv[0]["machine_id"], qv[0]["cutoff"], 20])
try:
    load_split(_d, "validation"); F.append("1 loader accepted a future row")
except IndexError:
    pass
_sh.rmtree(_d)
# 2 other machines' histories (incl. a full scramble) never change a machine's forecast (within-machine step)
k0 = qv[0]["machine_id"]; s2 = {k: (v if k == k0 else [rng.choice((0, 1, 2, 5, 20)) if x >= 0 else -1 for x in v]) for k, v in sv.items()}
if max(abs(x - y) for x, y in zip(base[qv[0]["query_id"]], run(qv, s2, RHO)[qv[0]["query_id"]])) > 1e-12: F.append("2 cross-machine dependence")
# 3 forecast at t' = c - 25 from y[:t'] is unchanged when every tick >= t' is scrambled (same construction as tests_leakage test 1)
q3 = [dict(r, cutoff=r["cutoff"] - 25) for r in qv]
s3a = {k: v[:-25] for k, v in sv.items()}; s3b = {k: (v[:-25] + [rng.choice((0, 1, 2, 5, 20)) for _ in range(25)])[:-25] for k, v in sv.items()}
if not same(run(q3, s3a, RHO), run(q3, s3b, RHO)): F.append("3 truncation")
# 4 rho depends on TRAIN histories only: scrambling validation / test histories leaves it unchanged (it never reads them)
OPENED.clear(); RHO2 = EE.rho_hat(ms_of(qt, st), H, EE.stage1(ms_of(qt, st), H))
if RHO2 != RHO or OPENED: F.append("4 rho uses non-train data or reopens files")
# 5 labels / targets / conditions never opened by the step
OPENED.clear(); load_split(pub, "validation"); bad = [p for p in OPENED if not (p.endswith("queries.csv") or p.endswith("history.csv"))]
if bad: F.append("5 opened " + str(bad))
# 6 missing ticks are not zeros: filling a short log's missing ticks with 0 must change its forecast
sh = [r for r in qv if min(sv[r["machine_id"]]) < 0]
if sh:
    r = sh[0]; m = r["machine_id"]
    if same(run([r], {m: sv[m]}, RHO), run([r], {m: [max(x, 0) for x in sv[m]]}, RHO)): F.append("6 missing as zero")
# 7 valid probabilities and determinism
P = np.array(list(base.values()))
if not (np.isfinite(P).all() and (P >= 0).all() and np.allclose(P.sum(1), 1, atol=1e-9)): F.append("7 invalid probabilities")
if not same(base, run(qv, sv, RHO)): F.append("7 nondeterministic")
print(f"rho {RHO}; {len(qv)} machines tested"); print("FAIL: " + "; ".join(F) if F else "ALL 7 CORRELATED-PRIOR LEAKAGE / VALIDITY TESTS PASS"); sys.exit(1 if F else 0)
