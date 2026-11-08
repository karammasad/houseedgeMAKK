"""Flip-time posterior calibration diagnostic (bounded). Production gate only (v8/v9 + shared-base gate), existing fitted worlds 1 and 2
(experiments/oracle/work{1,2}/gm.pkl), fresh held-out simulated test machines. Nothing is refit; production code untouched.
Posterior used: the production joint posterior over (structure, tau-grid + no flip), marginalized over structure."""
import math, pickle, sys, zlib
import numpy as np
sys.path.insert(0, "experiments/oracle")
import sim, arms
from adi import gate as G
from multiprocessing import get_context

_S = {}


def posterior(gm, ss, L, ty, sg):
    r = dict(machine_type=ty, signal=sg); cfg = gm._mcfg(ss, r_type=ty); taus, lp = G._taus(L, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(ss, L, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r), cfg["half_life"]); sp = gm._sp(ss)
    J, PR, TI = [], [], []
    for si, rows in enumerate(tb):
        if sp[si] <= 0: continue
        ls = math.log(sp[si])
        for ti, (ll, pr) in enumerate(rows): J.append(ll + ls + lp[ti]); PR.append(pr); TI.append(ti)
    J = np.array(J); w = np.exp(J - J.max()); w /= w.sum(); PR = np.array(PR); TI = np.array(TI); nt = len(taus)
    mt = np.bincount(TI, weights=w, minlength=nt + 1)                 # marginal over structure: [tau grid..., no flip]
    fl = TI < nt; Pf = float(w[fl].sum())
    pf = float((w[fl] * PR[fl]).sum() / Pf) if Pf > 1e-12 else float((w * PR).sum()); pn = float((w[~fl] * PR[~fl]).sum() / (1 - Pf)) if Pf < 1 - 1e-12 else float((w * PR).sum())
    return Pf, pf, pn, float((w * PR).sum()), np.array(taus), mt[:nt]


def job(m):
    g0 = _S["g0"]; Pf, pf, pn, p, taus, mt = posterior(g0, m["hist"], m["L"], m["type"], m["signal"]); F = m["tau"] is not None
    ptau = arms.pw(g0, m, tau="true")
    cov = mass = width = None
    if F and Pf > 1e-9:
        q = mt / mt.sum(); order = np.argsort(-q); cs = np.cumsum(q[order]); k = int(np.searchsorted(cs, 0.9)) + 1; S = set(order[:k].tolist())
        j = int(np.argmin(np.abs(taus - m["tau"]))); inside = taus[0] <= m["tau"] <= taus[-1] + 1
        cov = (j in S) if inside else None; mass = float(q[np.abs(taus - m["tau"]) <= 2].sum()) if inside else None
        width = int(taus[sorted(S)[-1]] - taus[sorted(S)[0]]) if len(S) else 0
    return dict(F=F, Pf=Pf, pf=pf, pn=pn, p=p, ptau=ptau, p_true=m["p_true"], cov=cov, mass=mass, width=width, short=m["short"], type=m["type"], ep=m["ep"], gated=m["pair"] is not None, tau=m["tau"], L=m["L"])


def real_check(gm):
    from common.data import public_dir, load_split
    q, seq = load_split(public_dir("../data/main"), "train"); out = []
    for r in q:
        s = seq[r["machine_id"]]
        if sum(v < 0 for v in s) < 0.1 * len(s): out.append(posterior(gm, s, len(s), r["machine_type"], r["signal"])[0])
    return np.array(out)


if __name__ == "__main__":
    res = []; N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    for seed in (1, 2):
        gm = pickle.load(open(f"experiments/oracle/work{seed}/gm.pkl", "rb")); w = sim.make_world(seed, n_test=N); _S["g0"] = gm
        with get_context("fork").Pool(8) as p: res += p.map(job, w["test"], chunksize=20)
    pickle.dump(res, open("experiments/oracle/flipcal_rows.pkl", "wb"))
    gm0 = pickle.load(open("experiments/gm_sb.pkl", "rb")); real = real_check(gm0)
    pickle.dump(real, open("experiments/oracle/flipcal_real.pkl", "wb")); print("done", len(res), len(real))
