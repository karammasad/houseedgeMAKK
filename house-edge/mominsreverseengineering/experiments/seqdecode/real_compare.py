"""Exact decoder vs local approximation on REAL held-out training machines (no validation file, no audit, no labels beyond the realized target).
Pseudo-queries: long TRAIN machines (<10% missing), target tick u in [45%, 58%) of the log (every 8th tick); history = ticks < u under the real masks
(n40 / short64 / n80 / full). IDENTICAL KNOWN PARAMETERS: per (machine, u) plug-in parameters (6 cell win rates, P(big|win), flip time) are fitted by
profile maximum likelihood on the machine's FULL pre-u history (an oracle, not deployable) and handed unchanged to BOTH predictors, so the only
difference is how the masked history is turned into a posterior over polarity/state. Five-category loss = squared error of
[1-p, p*sizes] with the production size vector (GateSizes + menu snap on the same masked history) -- identical for all predictors.
Reference column: production V0 gate (Beta-integrated parameters, fitted populations) on the same masked prefix, pc as in production.
Run (from mominsreverseengineering/):  PYTHONPATH=. python3 experiments/seqdecode/real_compare.py --data ../data/main
"""
import argparse, math, pickle, random, sys, time, zlib
from collections import defaultdict
from multiprocessing import get_context
import numpy as np
sys.path.insert(0, "experiments/seqdecode")
from common.data import public_dir, load_split
import prefixdiag as D
import prefixdiag_flipfair as F
from method4_ml.size_menu import snap
from hmm12 import Chain, exact_predict, local_predict

SIZES = (1, 2, 5, 20); MASKS = ("n40", "short64", "n80", "full"); PC = 0.5; _S = D._S


def cat(v): return -1 if v < 0 else (0 if v == 0 else (1 if v == 1 else 2))


def fit_plugin(c, u):
    """Profile-ML plug-in on the full pre-u cats. Returns Chain, tau_hat (or None), dLL vs one flat rate."""
    T = np.arange(3, u); c = np.asarray(c[:u]); g = ((c[T - 1] > 0) ^ (c[T - 3] > 0)).astype(int); c1 = c[T - 1]; w = (c[T] > 0).astype(float)
    cell = g * 3 + c1; n = np.zeros((len(T) + 1, 6)); k = np.zeros((len(T) + 1, 6))
    for i in range(len(T)):
        n[i + 1] = n[i]; k[i + 1] = k[i]; n[i + 1, cell[i]] += 1; k[i + 1, cell[i]] += w[i]
    def ll(kk, nn):
        r = (kk + 1) / (nn + 2); return float((kk * np.log(r) + (nn - kk) * np.log(1 - r)).sum()), r
    flip = lambda a: np.concatenate([a[3:], a[:3]])
    best = (ll(k[-1], n[-1])[0], None, ll(k[-1], n[-1])[1]); l0 = best[0]
    for tau in range(12, u - 5, 2):
        i = tau - 3; kk = k[i] + flip(k[-1] - k[i]); nn = n[i] + flip(n[-1] - n[i]); l, r = ll(kk, nn)
        if l > best[0] + 1e-9: best = (l, tau, r)
    if best[1] is not None and best[0] - l0 < 3.0: best = (l0, None, ll(k[-1], n[-1])[1])
    wt = w.sum(); flat = w.mean(); lflat = float(wt * math.log((wt + 1) / (len(w) + 2)) + (len(w) - wt) * math.log(1 - (wt + 1) / (len(w) + 2)))
    wins = c[T][w > 0]; beta = ((wins == 2).sum() + 1) / (len(wins) + 2); wb = (wt + 1) / (len(w) + 2)
    r = best[2].reshape(2, 3)
    return Chain(r, beta, [1 - wb, wb * (1 - beta), wb * beta]), best[1], best[0] - lflat


def job(args):
    r, = args; s = _S["seq"][r["machine_id"]]; L = len(s); gm, g, menu = _S["gm"], _S["g"], _S["menu"]; out = []
    cs = [cat(v) for v in s]
    for u in range(int(.45 * L), int(.58 * L), 8):
        ch, tau, dll = fit_plugin(cs, u); y = s[u]; onehot = np.array([int(y == 0)] + [int(y == z) for z in SIZES], float)
        for mask in MASKS:
            keep = D.keep_set(mask, u, s, r["machine_id"]); ss = D.masked(s, u, keep); obs = [cat(v) for v in ss]
            sz = np.array(snap(g.predict(r, ss), menu.get(r["machine_type"])))
            t0 = time.perf_counter(); pl = local_predict(ch, obs, u, PC); t1 = time.perf_counter(); pe, _ = exact_predict(ch, obs, u, PC); t2 = time.perf_counter()
            pv = min(max(F.v0_predict_pc(gm, r, u, ss), 1e-6), 1 - 1e-6)
            e5 = lambda pw: float(((np.concatenate([[1 - pw], pw * sz]) - onehot) ** 2).sum())
            out.append(dict(ep=r["episode_id"], mid=r["machine_id"], type=r["machine_type"], mask=mask, u=u, tau=tau, dll=dll, nmiss=int(sum(v < 0 for v in obs)),
                            e_loc=e5(1 - pl[0]), e_exa=e5(1 - pe[0]), e_v0=e5(pv), w_loc=(1 - pl[0] - (y > 0)) ** 2, w_exa=(1 - pe[0] - (y > 0)) ** 2,
                            dp=abs(pl[0] - pe[0]), t_loc=t1 - t0, t_exa=t2 - t1))
    return out


def cf(d, eps, reps=1500, seed=5):
    by = defaultdict(list)
    for x, e in zip(d, eps): by[e].append(x)
    keys = sorted(by); rng = random.Random(seed); S = {k: (sum(v), len(v)) for k, v in by.items()}; ms = []
    for _ in range(reps):
        ch = [S[rng.choice(keys)] for _ in keys]; ms.append(sum(a for a, _ in ch) / sum(b for _, b in ch))
    ms.sort(); return ms[int(.025 * reps)], ms[int(.975 * reps)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args(); pub = public_dir(a.data); t0 = time.time()
    from method4_ml.model import Model
    m = Model(cache=True, shared_base=True).fit(pub); g = m.size_models["gate"]; g.win_dir_lam = 0.0
    q, seq = load_split(pub, "train"); longs = [r for r in q if sum(v < 0 for v in seq[r["machine_id"]]) < 0.1 * len(seq[r["machine_id"]])]
    _S.update(seq=seq, gm=m.gm, g=g, menu=m.menu)
    with get_context("fork").Pool(8) as pool: res = pool.map(job, [(r,) for r in longs], chunksize=4)
    rows = [x for o in res for x in o]; pickle.dump(rows, open("experiments/seqdecode/real_rows.pkl", "wb"))
    print(f"{len(rows)} (target, mask) rows, {len(longs)} long machines, {len({r['ep'] for r in rows})} episodes, {time.time() - t0:.0f}s\n")
    def block(title, sel):
        print(title); print(f"  {'mask':8s}{'n':>6s}{'V0 prod':>9s}{'local':>9s}{'exact':>9s}{'exact-local x1e-4 [95% CI]':>34s}{'max|dp|':>10s}{'ms loc':>8s}{'ms exa':>8s}")
        for mk in MASKS:
            rs = [r for r in rows if r["mask"] == mk and sel(r)]
            if not rs: continue
            d = [(r["e_exa"] - r["e_loc"]) * 1e4 for r in rs]; lo, hi = cf(d, [r["ep"] for r in rs])
            print(f"  {mk:8s}{len(rs):6d}{np.mean([r['e_v0'] for r in rs]) * 1e4:9.1f}{np.mean([r['e_loc'] for r in rs]) * 1e4:9.1f}{np.mean([r['e_exa'] for r in rs]) * 1e4:9.1f}"
                  f"{np.mean(d):+12.4f} [{lo:+.4f},{hi:+.4f}]{max(r['dp'] for r in rs):10.2e}{np.mean([r['t_loc'] for r in rs]) * 1e3:8.2f}{np.mean([r['t_exa'] for r in rs]) * 1e3:8.2f}")
    print("five-category squared error x1e-4 (production sizes), paired by row"); block("ALL", lambda r: True)
    for ty in ("steady", "balanced", "jackpot"): block(ty, lambda r, ty=ty: r["type"] == ty)
    block("gate-explained (plug-in dLL vs flat >= 20)", lambda r: r["dll"] >= 20); block("weak gate (dLL < 20)", lambda r: r["dll"] < 20)
    block("plug-in says flipped before u", lambda r: r["tau"] is not None); block("plug-in says no flip", lambda r: r["tau"] is None)
    print("\nproduction V0 vs full: (V0 mask - V0 full) x1e-4 five-cat, the whole missing-history gap that ANY method could close")
    fl = {(r["mid"], r["u"]): r["e_v0"] for r in rows if r["mask"] == "full"}
    for mk in MASKS[:-1]:
        rs = [r for r in rows if r["mask"] == mk]; d = [(r["e_v0"] - fl[(r["mid"], r["u"])]) * 1e4 for r in rs]; lo, hi = cf(d, [r["ep"] for r in rs])
        print(f"  {mk:8s} {np.mean(d):+8.2f} [{lo:+.2f},{hi:+.2f}]")
    print(f"\ntotal {time.time() - t0:.0f}s")

if __name__ == "__main__": main()
