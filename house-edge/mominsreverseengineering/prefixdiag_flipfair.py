"""Flip-fair control for the prefix diagnostic (FINDINGS Entry 7). TRAIN HISTORIES ONLY; no audit, no validation.

Entry 7's comparison favored the hierarchical predictors: their window is flip-free by construction and they have no change-point model,
while v8 carries a flip prior (.5 when it classes the log as long, .05 when short). Here v8 gets the SAME no-flip assumption:
V0ff = v8's production gate with its flip prior p_change set to 1e-9 on every path (short and long). Nothing else changes (same cell priors,
same structure prior, same grids, same pseudo-queries / prefixes / targets / episode folds / scoring code as prefixdiag.py, imported unchanged).
Verification that flips are really off: V0ff is compared with a direct computation that passes NO flip times at all (taus = []).
Models on identical examples:  V0 (original v8)   V0ff (flip-fair v8)   HD (best existing hierarchical)   HA (extra column: no correlation term).
Run:  python3 prefixdiag_flipfair.py --data ../data/main
"""
import argparse, json, math, os, random, time
from collections import defaultdict
from multiprocessing import get_context
import numpy as np
from common.data import public_dir, load_split
from adi import gate as G
import hierarchy as H
import prefixdiag as D

PC_OFF = 1e-9
SPECS = {"HA": D.MODELS["HA"], "HD": D.MODELS["HD"]}
PREV = {"n10": (1478.6, -8.35), "n20": (936.6, -5.80), "n40": (880.4, -8.51), "n80": (884.1, -22.98), "short64": (891.3, -22.58), "full": (878.8, -22.78)}
_S = D._S


def fit_task(args):
    fold, name = args; d = _S["data"]; rows = d[0]
    idx = np.array([i for i, r in enumerate(rows) if fold is None or D.fold_of(r["ep"]) != fold])
    sub = ([rows[i] for i in idx], d[1][idx], d[2][idx], d[3][idx], d[4][idx])
    h = H.Hier(sub, **SPECS[name]); h.fit(maxiter=120)
    return fold, name, h.x.tolist(), h.names, h.ll, bool(h.res.success), int(h.res.nit)


def v0_predict_pc(gm, r, t, ss, pc=None, noflip=False):
    """Same arithmetic as prefixdiag.v0_predict; pc overrides the flip prior on this single path; noflip passes taus = [] (no flip times)."""
    cfg = gm._mcfg(ss, r_type=r["machine_type"])
    if pc is not None:
        cfg = dict(cfg, p_change=pc)
    if noflip:
        taus, lprior = [], [0.0]
    else:
        taus, lprior = G._taus(t, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(ss, t, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r), cfg["half_life"]); sp = gm._sp(ss)
    J, PR = [], []
    for si, rows in enumerate(tb):
        if sp[si] <= 0:
            continue
        ls = math.log(sp[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lprior[ti]); PR.append(pr)
    z = G.logsumexp(J); return sum(math.exp(j - z) * p for j, p in zip(J, PR))


def machine_job(args):
    r, = args; s = _S["seq"][r["machine_id"]]; L = len(s); fold = D.fold_of(r["episode_id"]); gm = _S["gm"]; P = _S["params"]; rows = []
    for t in range(int(.45 * L), int(.58 * L), 2):
        y = int(s[t] > 0); targ = D.targets_for(s, t)
        for mask in D.MASKS:
            ss = D.masked(s, t, D.keep_set(mask, t, s, r["machine_id"])); K, N, kt, nt = H.extract(ss, t)
            rec = dict(ep=r["episode_id"], mid=r["machine_id"], type=r["machine_type"], t=t, mask=mask, y=y,
                       V0=v0_predict_pc(gm, r, t, ss), V0ff=v0_predict_pc(gm, r, t, ss, pc=PC_OFF),
                       short_cls=sum(v < 0 for v in ss) > len(ss) / 2, pc_used=gm._mcfg(ss, r_type=r["machine_type"])["p_change"])
            if (t + len(mask)) % 11 == 0:                                   # verification sample (~9% of rows)
                rec["V0_noflip"] = v0_predict_pc(gm, r, t, ss, noflip=True)
            for name in SPECS:
                h, x = P[(fold, name)]; rec[name] = h.predict(x, r["machine_type"], r["signal"], K, N, kt, nt, targ)[0]
            rows.append(rec)
    return rows


def mean(v):
    return sum(v) / len(v)


def diff(rs, a, b):
    return [((r[a] - r["y"]) ** 2 - (r[b] - r["y"]) ** 2) * 1e4 for r in rs]


def fmt(rs, a, b):
    d = diff(rs, a, b); lo, hi = D.boot(d, [r["ep"] for r in rs]); return f"{mean(d):+7.2f} [{lo:+7.2f},{hi:+7.2f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    pub = public_dir(a.data); t0 = time.time()
    from method4_ml.model import Model
    _S["gm"] = Model(cache=True)._fit_gate(pub); _S["data"] = H.load_machines(pub); q, seq = load_split(pub, "train"); _S["seq"] = seq
    ctx = get_context("fork")
    with ctx.Pool(8) as pool:
        fits = pool.map(fit_task, [(f, m) for f in range(D.NF) for m in SPECS], chunksize=1)
    P = {}; flags = []
    for fold, name, x, names, ll, ok, nit in fits:
        h = H.Hier(_S["data"], **SPECS[name]); h.names = names; P[(fold, name)] = (h, np.array(x))
        flags.append(dict(fold=fold, model=name, converged=ok, iterations=nit, loglik=round(ll, 1)))
    json.dump(flags, open("experiments/flipfair_fit_flags.json", "w"), indent=1)
    print(f"cross-fits done in {time.time() - t0:.0f}s; nonconverged cross-fits: " +
          (", ".join(f"{f['model']} fold {f['fold']} ({f['iterations']} it)" for f in flags if not f["converged"]) or "none"), flush=True)
    _S["params"] = P
    long_rows = [r for r in q if sum(v < 0 for v in seq[r["machine_id"]]) < 0.1 * len(seq[r["machine_id"]])]
    with ctx.Pool(8) as pool:
        out = pool.map(machine_job, [(r,) for r in long_rows], chunksize=4)
    rows = [x for o in out for x in o]
    print(f"{len(rows)} rows in {time.time() - t0:.0f}s", flush=True)

    # (5) verification that the flip prior is disabled on the V0ff path
    ver = [r for r in rows if "V0_noflip" in r]
    gap = max(abs(r["V0ff"] - r["V0_noflip"]) for r in ver); gap_orig = mean([abs(r["V0"] - r["V0_noflip"]) for r in ver])
    print(f"\nFLIP-OFF VERIFICATION on {len(ver)} sampled rows: max |V0ff - (no flip times at all)| = {gap:.2e};  "
          f"mean |V0 - no-flip| = {gap_orig:.4f}  (original v8 does use flips)")
    pcs = defaultdict(set)
    for r in rows:
        pcs[r["mask"]].add(round(r["pc_used"], 3))
    print("original v8 flip prior by information setting (production rule: short if more than half the ticks are missing):",
          {m: sorted(v) for m, v in pcs.items()})
    sc = {m: mean([r["short_cls"] for r in rows if r["mask"] == m]) for m in D.MASKS}
    print("share classed SHORT by v8:", {m: round(v, 2) for m, v in sc.items()})

    rep = {m: [r for r in rows if r["mask"] == m] for m in D.MASKS}
    print("\nREPRODUCTION of Entry 7 (V0 Brier, HD - V0): " + ", ".join(
        f"{m} {mean([(r['V0'] - r['y']) ** 2 for r in rep[m]]) * 1e4:.1f}/{mean(diff(rep[m], 'HD', 'V0')):+.2f} (was {PREV[m][0]}/{PREV[m][1]:+.2f})" for m in D.MASKS))
    print("\nPAIRED squared-error differences vs REALIZED outcomes (x1e-4; negative = first model better; 95% CI clustered by episode).")
    print("Absolute levels include the Bernoulli floor; differences are unbiased for differences against the true probability.")
    print(f"{'info':8s} {'V0ff - V0':>24s} | {'HD - V0 (Entry 7)':>24s} | {'HD - V0ff (FAIR)':>24s} | {'HA - V0ff (FAIR)':>24s} | disappears")
    for m in D.MASKS:
        rs = rep[m]; d0 = mean(diff(rs, "HD", "V0")); d1 = mean(diff(rs, "HD", "V0ff"))
        gone = (1 - d1 / d0) * 100 if d0 < 0 else float("nan")
        print(f"{m:8s} {fmt(rs, 'V0ff', 'V0'):>24s} | {fmt(rs, 'HD', 'V0'):>24s} | {fmt(rs, 'HD', 'V0ff'):>24s} | {fmt(rs, 'HA', 'V0ff'):>24s} | {gone:5.0f}%")
    print("\nFAIR comparison HD - V0ff by machine type (x1e-4):")
    for m in ("n10", "n20", "n40", "short64", "full"):
        print(f"  {m:8s} " + "   ".join(f"{ty} {fmt([r for r in rep[m] if r['type'] == ty], 'HD', 'V0ff')}" for ty in ("balanced", "jackpot", "steady")))
    print("\nBrier levels (x1e-4): " + ", ".join(f"{m}: V0 {mean([(r['V0'] - r['y']) ** 2 for r in rep[m]]) * 1e4:.1f} V0ff {mean([(r['V0ff'] - r['y']) ** 2 for r in rep[m]]) * 1e4:.1f} HD {mean([(r['HD'] - r['y']) ** 2 for r in rep[m]]) * 1e4:.1f}" for m in D.MASKS))


if __name__ == "__main__":
    main()
