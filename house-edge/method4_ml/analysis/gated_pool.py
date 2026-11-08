"""E1/E2 (research campaign, 2026-11-08): context-gated pool of v8 and adi.cook, cross-fitted on the two kit validation halves.

E1 hypothesis: cook's gain (#39) is specific to short logs, so a FIXED 50/50 pool dilutes v8 on long logs; a 2-parameter pool
(w_short, w_long) chosen on one half generalizes to the other. Falsifier: out-of-fold gain <= 0 or paired bootstrap P < .9.
E2 hypothesis: v8-vs-cook disagreement in P(win) predicts v8's error (uncertainty signal). Falsifier: correlation ~ 0.
Reads validation targets (diagnosis + honest 2-fold cross-fit selection), never inside fit/predict.

  python -m method4_ml.analysis.gated_pool --data data/main
"""
import argparse, os, pickle, random, time
from collections import defaultdict
import numpy as np
from common.data import load_split
from common.evaluate import folds, points, _rows, COLS, CONDITIONS

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [round(x * .1, 1) for x in range(11)]


def get_preds(pub, workers):
    path = os.path.join(HERE, "..", "cache", "gated_pool_preds_" + os.path.basename(os.path.abspath(pub)) + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from method4_ml.cookmix.model import Model as CM
    t0 = time.time(); m = CM(workers=workers).fit(pub); print(f"fit cookmix parts {time.time() - t0:.0f}s", flush=True)
    from adi.cook import hier
    out = {}
    for sp in ("validation", "test"):
        hier.install(None); a = m.v8.predict(pub, sp); b = m.cook.predict(pub, sp); hier.install(None)
        out[sp] = (a, b)
    out["menu"] = m.v8.menu; out["short_obs"] = m.v8.gm.short_obs
    pickle.dump(out, open(path, "wb")); return out


def pool(a, b, q, seq, menu, short_obs, ws, wl):
    from method4_ml.size_menu import snap
    out = {}
    for r in q:
        k = r["query_id"]; short = sum(x >= 0 for x in seq[r["machine_id"]]) <= short_obs; w = ws if short else wl
        p = [w * x + (1 - w) * y for x, y in zip(a[k], b[k])]; pw = 1 - p[0]
        if menu is not None and pw > 0:
            s = snap([x / pw for x in p[1:]], menu.get(r["machine_type"])); p = [p[0]] + [pw * x for x in s]
        out[k] = p
    return out


def ep_err(pub, pred):
    """Per-episode mean squared error and its condition (for paired bootstrap of the contest metric)."""
    T = _rows(os.path.join(pub, "validation", "targets.csv")); C = {r["query_id"]: r["condition"] for r in _rows(os.path.join(pub, "validation", "conditions.csv"))}
    e = defaultdict(list); c = {}
    for t in T:
        p = pred[t["query_id"]]; e[t["episode_id"]].append(sum((x - float(t[k])) ** 2 for x, k in zip(p, COLS))); c[t["episode_id"]] = C[t["query_id"]]
    return {k: float(np.mean(v)) for k, v in e.items()}, c


def boot_points(pub, eA, eB, cond, eps, n=2000, seed=0):
    """Paired bootstrap over episodes within condition: P(points(B) > points(A)). Reference errors are fixed per episode."""
    R = _rows(os.path.join(pub, "validation", "reference.csv")); T = _rows(os.path.join(pub, "validation", "targets.csv"))
    ref = {r["query_id"]: [float(r[k]) for k in COLS] for r in R}; er = defaultdict(list)
    for t in T:
        er[t["episode_id"]].append(sum((x - float(t[k])) ** 2 for x, k in zip(ref[t["query_id"]], COLS)))
    er = {k: np.mean(v) for k, v in er.items()}; rng = random.Random(seed); by = defaultdict(list)
    for e in eps:
        by[cond[e]].append(e)
    wins = 0
    for _ in range(n):
        RA = RB = 0
        for c in CONDITIONS:
            s = [rng.choice(by[c]) for _ in by[c]]; b = np.mean([er[e] for e in s])
            RA += np.mean([eA[e] for e in s]) / b; RB += np.mean([eB[e] for e in s]) / b
        wins += RB < RA
    return wins / n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--workers", type=int, default=None)
    a_ = ap.parse_args(); pub = a_.data
    P = get_preds(pub, a_.workers); a, b = P["validation"]; q, seq = load_split(pub, "validation")
    tune, audit = folds(pub); menu, so = P["menu"], P["short_obs"]
    v8 = pool(a, b, q, seq, menu, so, 1.0, 1.0); fixed = pool(a, b, q, seq, menu, so, .5, .5)
    print(f"v8         tuning {points(pub, v8, tune)[0]:.3f} audit {points(pub, v8, audit)[0]:.3f} full {points(pub, v8)[0]:.3f}")
    print(f"cook       full {points(pub, b)[0]:.3f}")
    print(f"fixed .5   tuning {points(pub, fixed, tune)[0]:.3f} audit {points(pub, fixed, audit)[0]:.3f} full {points(pub, fixed)[0]:.3f}")
    # E1: 2-fold cross-fit of (w_short, w_long); also 1-parameter global w for the ablation
    oof = {}; oof1 = {}
    for sel, ev, name in ((tune, audit, "tune->audit"), (audit, tune, "audit->tune")):
        sc = {(ws, wl): points(pub, pool(a, b, q, seq, menu, so, ws, wl), sel)[0] for ws in GRID for wl in GRID}
        ws, wl = max(sc, key=sc.get); g = max(GRID, key=lambda w: sc[(w, w)])
        pe = pool(a, b, q, seq, menu, so, ws, wl); pg = pool(a, b, q, seq, menu, so, g, g)
        print(f"{name}: chose w_short {ws} w_long {wl} (sel {sc[(ws, wl)]:.3f}); held-out {points(pub, pe, ev)[0]:.3f} vs v8 {points(pub, v8, ev)[0]:.3f} vs fixed {points(pub, fixed, ev)[0]:.3f}; global w {g}: held-out {points(pub, pg, ev)[0]:.3f}")
        ids = {r["query_id"] for r in q if r["episode_id"] in ev}
        oof.update({k: pe[k] for k in ids}); oof1.update({k: pg[k] for k in ids})
    eV, cond = ep_err(pub, v8); eO, _ = ep_err(pub, oof); eF, _ = ep_err(pub, fixed); eG, _ = ep_err(pub, oof1)
    eps = sorted(eV)
    print(f"\nOUT-OF-FOLD full: gated {points(pub, oof)[0]:.3f}  global-w {points(pub, oof1)[0]:.3f}  fixed .5 {points(pub, fixed)[0]:.3f}  v8 {points(pub, v8)[0]:.3f}")
    print(f"paired bootstrap P(better than v8): gated {boot_points(pub, eV, eO, cond, eps):.3f}  fixed .5 {boot_points(pub, eV, eF, cond, eps):.3f}  global-w {boot_points(pub, eV, eG, cond, eps):.3f}")
    print(f"P(gated better than fixed .5): {boot_points(pub, eF, eO, cond, eps):.3f}")
    for c in CONDITIONS:
        es = [e for e in eps if cond[e] == c]
        print(f"  {c:14s} episodes improved by fixed .5: {sum(eF[e] < eV[e] for e in es)}/{len(es)}   mean err v8 {np.mean([eV[e] for e in es]):.5f} fixed {np.mean([eF[e] for e in es]):.5f} gated {np.mean([eO[e] for e in es]):.5f}")
    # E2: disagreement vs v8 error
    T = {r["query_id"]: [float(r[k]) for k in COLS] for r in _rows(os.path.join(pub, "validation", "targets.csv"))}
    d = np.array([abs(a[k][0] - b[k][0]) for k in T]); err = np.array([sum((x - y) ** 2 for x, y in zip(v8[k], T[k])) for k in T])
    gain = np.array([sum((x - y) ** 2 for x, y in zip(v8[k], T[k])) - sum((x - y) ** 2 for x, y in zip(fixed[k], T[k])) for k in T])
    from scipy.stats import spearmanr
    print(f"\nE2 disagreement |dP(win)|: spearman with v8 error {spearmanr(d, err)[0]:.3f}; with pooling gain {spearmanr(d, gain)[0]:.3f}; "
          f"top-decile disagreement holds {gain[d >= np.quantile(d, .9)].sum() / max(gain.sum(), 1e-12):.0%} of the pooling gain")


if __name__ == "__main__":
    main()
