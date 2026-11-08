"""E5 (2026-11-08): conservative probability-space corrections of the v8 champion, cross-fitted on the kit halves.

Hypothesis: v8 over- or under-personalizes machines with few observations, so its final 5-class vector should move toward a
robust pooled prior with a strength that depends on the effective sample size (observed ticks). Ablations:
  a  shrink toward pooled per-type 5-class frequency (train histories), lambda = c / (c + n_obs / 64), one c
  b  same with separate c for short / long logs
  c  log-space temperature p ~ p**tau, separate tau for short / long (sharpen < 1 < flatten)
  d  shrink the SIZE part only (given a win) toward the type's pooled size distribution, one strength per type
Each has 1-3 parameters picked on one validation half and scored on the other (both directions), paired episode bootstrap.
Falsifier: out-of-fold gain <= 0 or P(better) < .9. v8 predictions come from analysis/gated_pool's cache (no refit).
Reads validation targets (diagnosis + cross-fit selection), never inside fit/predict.
"""
import itertools, os, pickle
from collections import defaultdict
import numpy as np
from common.data import load_split, PAYOUTS
from common.evaluate import folds, points, CONDITIONS
from method4_ml.analysis.gated_pool import ep_err, boot_points

PUB = "data/main"
C_GRID = (0, .05, .1, .2, .4, .8, 1.6)
TAU_GRID = (.85, .9, .95, 1.0, 1.05, 1.1, 1.2)
S_GRID = (0, 2, 5, 10, 20, 40)


def pooled_prior(pub):
    q, seq = load_split(pub, "train"); c = defaultdict(lambda: np.ones(5))
    for r in q:
        for x in seq[r["machine_id"]]:
            if x >= 0:
                c[r["machine_type"]][PAYOUTS.index(x)] += 1
    return {t: v / v.sum() for t, v in c.items()}


def apply(v8, q, seq, prior, kind, prm, so):
    out = {}
    for r in q:
        k = r["query_id"]; p = np.array(v8[k]); obs = sum(x >= 0 for x in seq[r["machine_id"]]); short = obs <= so
        pi = prior.get(r["machine_type"], np.full(5, .2))
        if kind == "a":
            c = prm[0]; lam = c / (c + obs / 64) if c > 0 else 0; p = (1 - lam) * p + lam * pi
        elif kind == "b":
            c = prm[0] if short else prm[1]; lam = c / (c + obs / 64) if c > 0 else 0; p = (1 - lam) * p + lam * pi
        elif kind == "c":
            tau = prm[0] if short else prm[1]; p = np.clip(p, 1e-12, 1) ** tau; p /= p.sum()
        elif kind == "d":
            s = prm[{"steady": 0, "balanced": 1, "jackpot": 2}.get(r["machine_type"], 0)]
            pw = 1 - p[0]
            if pw > 0 and s > 0:
                sz = p[1:] / pw; ps = pi[1:] / pi[1:].sum(); n = sum(x > 0 for x in seq[r["machine_id"]])
                sz = (n * sz + s * ps) / (n + s); p = np.r_[p[0], pw * sz]
        out[k] = (p / p.sum()).tolist()
    return out


def grid(kind):
    return {"a": [(c,) for c in C_GRID], "b": list(itertools.product(C_GRID, C_GRID)),
            "c": list(itertools.product(TAU_GRID, TAU_GRID)), "d": list(itertools.product(S_GRID, S_GRID, S_GRID))}[kind]


def main():
    P = pickle.load(open(os.path.join(os.path.dirname(__file__), "..", "cache", "gated_pool_preds_main.pkl"), "rb"))
    v8 = P["validation"][0]; so = P["short_obs"]; q, seq = load_split(PUB, "validation"); tune, audit = folds(PUB)
    prior = pooled_prior(PUB); eV, cond = ep_err(PUB, v8); eps = sorted(eV)
    print(f"v8 champion: tuning {points(PUB, v8, tune)[0]:.3f} audit {points(PUB, v8, audit)[0]:.3f} full {points(PUB, v8)[0]:.3f}")
    print(f"  ratios {dict((c, round(r, 5)) for c, r in points(PUB, v8)[1].items())}")
    for kind in "abcd":
        oof = {}; picks = []
        for sel, ev in ((tune, audit), (audit, tune)):
            ids = {r["query_id"] for r in q if r["episode_id"] in ev}
            # selection uses predictions restricted to the selection half only
            best = max(grid(kind), key=lambda prm: points(PUB, apply(v8, q, seq, prior, kind, prm, so), sel)[0])
            picks.append(best); pe = apply(v8, q, seq, prior, kind, best, so); oof.update({k: pe[k] for k in ids})
        eO, _ = ep_err(PUB, oof); pts, rat = points(PUB, oof)
        dif = {c: np.mean([eO[e] - eV[e] for e in eps if cond[e] == c]) for c in CONDITIONS}
        print(f"\n[{kind}] picks tune->audit {picks[0]}  audit->tune {picks[1]}")
        print(f"    OUT-OF-FOLD full {pts:.3f} (v8 {points(PUB, v8)[0]:.3f}, delta {pts - points(PUB, v8)[0]:+.4f}); "
              f"P(better than v8) {boot_points(PUB, eV, eO, cond, eps):.3f}")
        print("    per-condition mean episode error change: " + ", ".join(f"{c} {d:+.2e}" for c, d in dif.items()))


if __name__ == "__main__":
    main()
