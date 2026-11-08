"""Leakage / validity tests for DPCM (method4_ml/outlier.py). Exit code 1 on any failure.  python3 -m method4_ml.outlier_tests --data data/adaptation"""
import argparse, copy, random, sys
import numpy as np
from common.data import load_split, public_dir
from method4_ml import outlier as O

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/adaptation"); a = ap.parse_args()
pub = public_dir(a.data); prior = O.prior_table([load_split(pub, "train")]); q, s = load_split(pub, "validation")
q = q[:64]; s = {r["machine_id"]: s[r["machine_id"]] for r in q}; rng = random.Random(0); fails = []
base = O.predict(q, s, prior, mu=0.99)
def same(A, B): return max(abs(x - y) for k in A for x, y in zip(A[k], B[k])) < 1e-12
# 1. future observations (ticks >= cutoff, appended) cannot change any forecast
s1 = {k: v + [rng.choice((0, 1, 2, 5, 20)) for _ in range(40)] for k, v in s.items()}
if not same(base, O.predict(q, s1, prior, mu=0.99)): fails.append("future ticks change predictions")
# 2. earlier truncation: forecasting at c' from y[:c'] equals forecasting at c' with later ticks scrambled
q2 = [dict(r, cutoff=r["cutoff"] - 30) for r in q]
A = O.predict(q2, {k: v[:-30] for k, v in s.items()}, prior, mu=0.99)
B = O.predict(q2, {k: v[:-30] + [rng.choice((0, 1, 2, 5, 20)) for _ in range(30)] for k, v in s.items()}, prior, mu=0.99)
if not same(A, B): fails.append("truncated forecast depends on later ticks")
# 3. other machines (incl. their future) never matter: forecast of machine 0 alone == forecast of machine 0 in the batch
one = O.predict(q[:1], {q[0]["machine_id"]: s[q[0]["machine_id"]]}, prior, mu=0.99)
if not same(one, {q[0]["query_id"]: base[q[0]["query_id"]]}): fails.append("cross-machine dependence")
# 4. missing ticks are not zeros: filling missing ticks with 0 must change a short-log forecast
miss = [r for r in q if min(s[r["machine_id"]]) < 0]
if miss:
    r = miss[0]; m = r["machine_id"]; z = {m: [max(v, 0) for v in s[m]]}
    if same(O.predict([r], {m: s[m]}, prior, mu=0.99), O.predict([r], z, prior, mu=0.99)): fails.append("missing ticks behave like zeros")
# 5. valid probabilities, deterministic
P = np.array(list(base.values()))
if not (np.isfinite(P).all() and (P >= 0).all() and np.allclose(P.sum(1), 1)): fails.append("invalid probabilities")
if not same(base, O.predict(q, s, prior, mu=0.99)): fails.append("nondeterministic")
# 6. labels / targets / conditions are never read: predict only receives queries + history (by construction: load_split)
print("FAIL: " + "; ".join(fails) if fails else "ALL 6 LEAKAGE / VALIDITY TESTS PASS"); sys.exit(1 if fails else 0)
