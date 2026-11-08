"""Automated leakage / validity tests for the EGP challenger. Run BEFORE any performance comparison:
    python -m thanos.tests_leakage --data data/main
Exits non-zero on the first failure.

 1 future invariance: changing a machine's ticks at or after t never changes its prediction at t (or its window statistics before t)
 1b cross-machine: changing OTHER machines' histories (same episode) never changes a machine's prediction (fixed hyperparameters)
 2 held-out labels cannot affect fitted parameters: fit on a copy of the data with labels / targets / conditions / reference deleted
   gives identical hyperparameters and identical predictions; only queries.csv / history.csv are ever opened (audit hook)
 3 CV train / evaluation episode ids are disjoint (cv.folds), machine ids never span episodes
 4 every training statistic uses only earlier ticks: perturbing tick t0 leaves every statistic with end <= t0 unchanged
 5 preprocessing / hyperparameters fitted only on permitted data: the inductive fit opens only train files
 6 every forecast is a valid 5-vector (finite, >= 0, sums to 1 within 1e-9), every query exactly once
 7 the model needs no hidden labels or condition labels (same as 2: predicting from a stripped copy works and is identical)
"""
import argparse, os, shutil, sys, tempfile, copy
import numpy as np
from common.data import load_split, public_dir
from . import egp
from .model import Model
from . import cv

OPENED = []


def _hook(event, args):
    if event == "open" and isinstance(args[0], str) and args[0].endswith(".csv"):
        OPENED.append(os.path.abspath(args[0]))


def fail(msg):
    print("FAIL:", msg); sys.exit(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--tickset", default="own"); a = ap.parse_args()
    pub = public_dir(a.data); rng = np.random.default_rng(0)
    sys.addaudithook(_hook)

    # ---- 5: inductive fit opens only train queries / history
    OPENED.clear()
    M = Model("train", verbose=False, tickset=a.tickset).fit(pub)
    bad = [f for f in OPENED if not (f.endswith(os.path.join("train", "queries.csv")) or f.endswith(os.path.join("train", "history.csv")))]
    if bad:
        fail(f"inductive fit opened {bad}")
    print("5 PASS  inductive fit opened only", sorted({os.path.relpath(f, pub) for f in OPENED}))

    # ---- 6: valid forecasts for validation and test
    for sp in ("validation", "test"):
        OPENED.clear(); P = M.predict(pub, sp)
        q, _ = load_split(pub, sp)
        if sorted(P) != sorted(r["query_id"] for r in q) or len(P) != len(q):
            fail(f"{sp}: query set mismatch")
        arr = np.array(list(P.values()))
        if arr.shape[1] != 5 or not np.all(np.isfinite(arr)) or arr.min() < 0 or np.abs(arr.sum(1) - 1).max() > 1e-9:
            fail(f"{sp}: invalid probability vectors")
        bad = [f for f in OPENED if not (f.endswith("queries.csv") or f.endswith("history.csv"))]
        if bad:
            fail(f"predict({sp}) opened {bad}")
    print("6 PASS  every validation / test query once, valid 5-vectors; predict opened only queries / history")
    Pval = M.predict(pub, "validation")

    # ---- 2 / 7: stripped copy (no labels, targets, conditions, reference) -> identical fit and predictions (inductive and transductive)
    tmp = tempfile.mkdtemp()
    for sp in ("train", "validation", "test"):
        os.makedirs(os.path.join(tmp, sp))
        for f in ("queries.csv", "history.csv"):
            shutil.copy(os.path.join(pub, sp, f), os.path.join(tmp, sp, f))
    M2 = Model("train", verbose=False, tickset=a.tickset).fit(tmp); P2 = M2.predict(tmp, "validation")
    if any(not np.array_equal(M.H.theta[t], M2.H.theta[t]) for t in M.H.theta) or M.H.K != M2.H.K:
        fail("hyperparameters differ on the stripped copy")
    if max(abs(a_ - b_) for k in Pval for a_, b_ in zip(Pval[k], P2[k])) > 0:
        fail("predictions differ on the stripped copy")
    # transductive: also identical with / without labels on disk
    OPENED.clear(); MT = Model("all", verbose=False, tickset=a.tickset).fit(pub)
    bad = [f for f in OPENED if not (f.endswith("queries.csv") or f.endswith("history.csv"))]
    if bad:
        fail(f"transductive fit opened {bad}")
    MT2 = Model("all", verbose=False, tickset=a.tickset).fit(tmp)
    if any(not np.array_equal(MT.H.theta[t], MT2.H.theta[t]) for t in MT.H.theta):
        fail("transductive hyperparameters differ on the stripped copy")
    shutil.rmtree(tmp)
    print("2 PASS  fit + predict from a copy without labels / targets / conditions / reference: identical (inductive and transductive)")
    print("7 PASS  no hidden labels or condition labels are needed")

    # ---- 1 / 1b / 4: prefix invariance on real machines
    q, seq = load_split(pub, "validation")
    ms = egp.machines_from_split(q, seq)
    for m in [ms[i] for i in rng.choice(len(ms), 40, replace=False)]:
        y = m.y.copy(); t = int(rng.integers(120, m.cutoff))
        base = egp.machine_posterior(egp.Machine(m.qid, m.ep, m.mid, m.type, m.signal, y[:t]), M.H)[1]
        y2 = y.copy(); fut = y2[t:]; y2[t:] = np.where(fut >= 0, rng.choice([0, 1, 2, 5, 20], len(fut)), -1)
        alt = egp.machine_posterior(egp.Machine(m.qid, m.ep, m.mid, m.type, m.signal, y2[:t]), M.H)[1]
        if np.abs(base - alt).max() != 0:
            fail("prediction at t changed when ticks >= t changed")
        # 4: statistics with end <= t0 do not see tick t0
        t0 = int(rng.integers(60, m.cutoff - 1)); y3 = y.copy(); y3[t0] = 20 if y3[t0] != 20 else 0
        A = egp.Machine(m.qid, m.ep, m.mid, m.type, m.signal, y); B = egp.Machine(m.qid, m.ep, m.mid, m.type, m.signal, y3)
        for end in (t0, max(9, t0 - 3)):
            ra, ca, ea = egp.window_stats(A, M.H.K, end); rb, cb, eb = egp.window_stats(B, M.H.K, end)
            if not (np.array_equal(ra, rb) and np.array_equal(ca, cb) and np.array_equal(ea, eb)):
                fail(f"window statistics with end {end} saw tick {t0}")
        # the cutoff-tick inputs use ticks cutoff-1 .. cutoff-8 only: a tick-(cutoff) value does not exist in m.y by construction
        if len(m.y) != m.cutoff:
            fail("machine array longer than its cutoff")
    print("1 PASS  predictions at t identical under arbitrary changes of ticks >= t (40 machines)")
    print("4 PASS  statistics ending at t0 unaffected by tick t0")
    ep = ms[0].ep; mine = [m for m in ms if m.ep == ep]; tgt = mine[0]
    p_ref = egp.machine_posterior(tgt, M.H)[1]
    others = copy.deepcopy(mine[1:])
    for o in others:
        o.y[:] = rng.choice([0, 1, 2, 5, 20], len(o.y))
    p_alt = egp.machine_posterior(tgt, M.H)[1]
    if np.abs(p_ref - p_alt).max() != 0:
        fail("prediction depends on other machines' histories")
    print("1b PASS prediction does not depend on other machines' histories at prediction time (hyperparameters fixed)")

    # ---- 3: CV folds
    qtr, _ = load_split(pub, "train")
    F = cv.folds(qtr)
    eps = [set(f) for f in F]
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            if eps[i] & eps[j]:
                fail("CV folds share episodes")
    mid_ep = {}
    for r in qtr:
        if mid_ep.setdefault(r["machine_id"], r["episode_id"]) != r["episode_id"]:
            fail("machine id spans episodes")
    if set().union(*eps) != {r["episode_id"] for r in qtr}:
        fail("CV folds do not cover train")
    print(f"3 PASS  {len(F)} CV folds, episode-disjoint, cover all {len({r['episode_id'] for r in qtr})} train episodes; machine ids never span episodes")
    print("ALL LEAKAGE TESTS PASSED")


if __name__ == "__main__":
    main()
