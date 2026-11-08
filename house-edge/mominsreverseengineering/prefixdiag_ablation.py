"""Size-shift ablation (FINDINGS Entry 8 follow-up). TRAIN HISTORIES ONLY; no audit, no validation.

Same cross-fitting folds, pseudo-queries, prefixes, targets and flip-fair v8 baseline as prefixdiag_flipfair.py (its helpers are imported unchanged).
Models on identical examples:
  V0ff  flip-fair v8 (flip prior 1e-9, verified off in Entry 8)
  HD    existing hierarchical predictor (shared base rate per side, size shift gamma learned per type, on/off correlation, shared sigma / slopes)
  HDg0  HD with gamma FIXED TO ZERO; everything else identical (pooling, lag support, fitting procedure, folds, starting values, 120-iteration cap)
Note on interpretation: with gamma = 0 the six size-split sub-cells share one rate per side, so HDg0 is a pooled plain-XOR predictor with a tight
population prior and the hierarchy-fit structure weights (3 pairs + none).
Run:  python3 prefixdiag_ablation.py --data ../data/main
"""
import argparse, json, time
from multiprocessing import get_context
import numpy as np
from common.data import public_dir, load_split
import hierarchy as H
import prefixdiag as D
import prefixdiag_flipfair as F

SPECS = {"HD": D.MODELS["HD"], "HDg0": dict(D.MODELS["HD"], gam_fixed=0.0)}
_S = D._S


def fit_task(args):
    fold, name = args; d = _S["data"]; rows = d[0]
    idx = np.array([i for i, r in enumerate(rows) if fold is None or D.fold_of(r["ep"]) != fold])
    sub = ([rows[i] for i in idx], d[1][idx], d[2][idx], d[3][idx], d[4][idx])
    h = H.Hier(sub, **SPECS[name]); h.fit(maxiter=120)
    return fold, name, h.x.tolist(), h.names, float(h.ll), bool(h.res.success), int(h.res.nit), str(h.res.message)


def machine_job(args):
    r, = args; s = _S["seq"][r["machine_id"]]; L = len(s); fold = D.fold_of(r["episode_id"]); gm = _S["gm"]; P = _S["params"]; rows = []
    for t in range(int(.45 * L), int(.58 * L), 2):
        y = int(s[t] > 0); targ = D.targets_for(s, t)
        for mask in D.MASKS:
            ss = D.masked(s, t, D.keep_set(mask, t, s, r["machine_id"])); K, N, kt, nt = H.extract(ss, t)
            rec = dict(ep=r["episode_id"], type=r["machine_type"], t=t, mask=mask, y=y,
                       V0ff=F.v0_predict_pc(gm, r, t, ss, pc=F.PC_OFF))
            for name in SPECS:
                h, x = P[(fold, name)]; rec[name] = h.predict(x, r["machine_type"], r["signal"], K, N, kt, nt, targ)[0]
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    pub = public_dir(a.data); t0 = time.time()
    from method4_ml.model import Model
    _S["gm"] = Model(cache=True)._fit_gate(pub); _S["data"] = H.load_machines(pub); q, seq = load_split(pub, "train"); _S["seq"] = seq
    ctx = get_context("fork")
    with ctx.Pool(8) as pool:
        fits = pool.map(fit_task, [(f, m) for f in list(range(D.NF)) + [None] for m in SPECS], chunksize=1)
    P = {}; status = []
    for fold, name, x, names, ll, ok, nit, msg in fits:
        h = H.Hier(_S["data"], **SPECS[name]); h.names = names
        if fold is not None:
            P[(fold, name)] = (h, np.array(x))
        status.append(dict(fold="all" if fold is None else fold, model=name, converged=ok, iterations=nit, loglik=round(ll, 1), message=msg))
        if name == "HDg0":
            g = [h.get(np.array(x), "gam", t) for t in H.TYPES]; assert max(abs(v) for v in g) < 1e-9, g
    json.dump(status, open("experiments/ablation_fit_status.json", "w"), indent=1)
    print(f"fits done in {time.time() - t0:.0f}s\n\nOPTIMIZER STATUS (L-BFGS-B, 120-iteration cap; gamma of HDg0 verified = 0 in every fit):")
    print(f"  {'fold':5s} {'model':5s} {'converged':>9s} {'iters':>6s} {'loglik':>10s}")
    for s_ in sorted(status, key=lambda z: (str(z["fold"]), z["model"])):
        print(f"  {str(s_['fold']):5s} {s_['model']:5s} {str(s_['converged']):>9s} {s_['iterations']:6d} {s_['loglik']:10.1f}")
    ll = {s_["model"]: s_["loglik"] for s_ in status if s_["fold"] == "all"}
    print(f"  in-sample (full data): HD {ll['HD']}  HDg0 {ll['HDg0']}  -> gamma adds {ll['HD'] - ll['HDg0']:+.1f} log-lik for 3 parameters", flush=True)
    _S["params"] = P
    long_rows = [r for r in q if sum(v < 0 for v in seq[r["machine_id"]]) < 0.1 * len(seq[r["machine_id"]])]
    with ctx.Pool(8) as pool:
        out = pool.map(machine_job, [(r,) for r in long_rows], chunksize=4)
    rows = [x for o in out for x in o]; print(f"\n{len(rows)} rows in {time.time() - t0:.0f}s")
    rep = {m: [r for r in rows if r["mask"] == m] for m in D.MASKS}
    print("\nREPRODUCTION of Entry 8, HD - V0ff: " + ", ".join(f"{m} {F.mean(F.diff(rep[m], 'HD', 'V0ff')):+.2f}" for m in D.MASKS)
          + "   (Entry 8: n10 -8.07, n20 -4.44, n40 -6.96, n80 -8.26, short64 -8.03, full -8.62)")
    print("\nPAIRED squared-error differences vs realized outcomes (x1e-4, negative = first model better, 95% CI clustered by episode):")
    print(f"{'info':8s} {'HD - V0ff':>25s} | {'HDg0 - V0ff':>25s} | {'HDg0 - HD':>25s}")
    for m in D.MASKS:
        print(f"{m:8s} {F.fmt(rep[m], 'HD', 'V0ff'):>25s} | {F.fmt(rep[m], 'HDg0', 'V0ff'):>25s} | {F.fmt(rep[m], 'HDg0', 'HD'):>25s}")
    for ty in ("jackpot", "balanced", "steady"):
        print(f"\n{ty.upper()} machines:")
        for m in ("n10", "n20", "n40", "short64", "full"):
            rs = [r for r in rep[m] if r["type"] == ty]
            print(f"  {m:8s} n={len(rs):4d} {F.fmt(rs, 'HD', 'V0ff'):>25s} | {F.fmt(rs, 'HDg0', 'V0ff'):>25s} | {F.fmt(rs, 'HDg0', 'HD'):>25s}")
    print("\nBrier levels (x1e-4): " + ", ".join(f"{m}: V0ff {F.mean([(r['V0ff'] - r['y']) ** 2 for r in rep[m]]) * 1e4:.1f} HD "
          f"{F.mean([(r['HD'] - r['y']) ** 2 for r in rep[m]]) * 1e4:.1f} HDg0 {F.mean([(r['HDg0'] - r['y']) ** 2 for r in rep[m]]) * 1e4:.1f}" for m in D.MASKS))


if __name__ == "__main__":
    main()
