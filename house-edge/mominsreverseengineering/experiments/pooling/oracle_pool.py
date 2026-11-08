"""Bounded test: how much could pooling the shared flip time across the 8 machines of a shared_change episode help? TRAIN ONLY, gate model gm_sb, no refit.
No true flip times exist in train, so the 'oracle' tau_ep = median over the episode's machines of the production full-history flip estimate (within-episode std ~1 tick):
a generous stand-in that uses ticks AFTER the target. Pooled = reweight each machine's own joint (structure, tau) posterior by the product of the OTHER machines'
tau-likelihood ratios computed from ticks BEFORE the target only (what a real pooling method could use). Targets: realized payout of pseudo-queries; 5-cat squared error
with the production size vector."""
import math, pickle, sys
from collections import defaultdict
import numpy as np
from common.data import public_dir, load_split
from adi import gate as G
from multiprocessing import get_context

_S = {}


def joint(gm, ss, cut, ty, sg, taus_override=None):
    r = dict(machine_type=ty, signal=sg); cfg = gm._mcfg(ss, r_type=ty)
    taus, lp = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"]) if taus_override is None else taus_override
    tb = G.machine_tables(ss, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r), cfg["half_life"]); sp = gm._sp(ss)
    J, PR, TI = [], [], []
    for si, rows in enumerate(tb):
        if sp[si] <= 0: continue
        for ti, (ll, pr) in enumerate(rows): J.append(ll + math.log(sp[si]) + lp[ti]); PR.append(pr); TI.append(ti)
    J = np.array(J); w = np.exp(J - J.max()); w /= w.sum(); return w, np.array(PR), np.array(TI), len(taus), np.exp(np.array(lp))


def job(args):
    ep, u, tau_ep = args; gm = _S["gm"]; ms = _S["eps"][ep]; out = []; J = []
    for r in ms:
        s = _S["seq"][r["machine_id"]]; ss = list(s[:u]); J.append(joint(gm, ss, u, r["machine_type"], r["signal"]))
    lls = []                                                        # log tau-likelihood ratio post/prior per machine over [taus..., None]
    for w, PR, TI, nt, pr in J:
        post = np.bincount(TI, weights=w, minlength=nt + 1); lls.append(np.log(np.maximum(post, 1e-300) / np.maximum(pr, 1e-300)))
    tot = np.sum(lls, 0)
    for k, r in enumerate(ms):
        s = _S["seq"][r["machine_id"]]; ss = list(s[:u]); w, PR, TI, nt, pr = J[k]
        p_prod = float((w * PR).sum())
        oth = tot - lls[k]; w2 = w * np.exp(oth[TI] - oth[TI].max()); w2 /= w2.sum(); p_pool = float((w2 * PR).sum())
        taus_o = ([], [0.0]) if u <= tau_ep else ([tau_ep], [0.0, -1e300])
        wo, PRo, _, _, _ = joint(gm, ss, u, r["machine_type"], r["signal"], taus_o); p_or = float((wo * PRo).sum())
        out.append(dict(ep=ep, mid=r["machine_id"], u=u, rel=u - tau_ep, L=len(s), type=r["machine_type"], p_prod=p_prod, p_pool=p_pool, p_or=p_or))
    return out


def main():
    pub = public_dir("../data/main"); q, seq = load_split(pub, "train"); gm = pickle.load(open("experiments/gm_sb.pkl", "rb"))
    R = pickle.load(open("experiments/forensics/rows.pkl", "rb")); P = [r for r in R if r["kind"] == "pseudo" and r["cond"] == "shared_change"]
    tau = defaultdict(list)
    for m in {r["mid"]: r for r in P}.values():
        if m["tau"] is not None: tau[m["ep"]].append(m["tau"])
    tau_ep = {e: float(np.median(v)) for e, v in tau.items()}
    eps = defaultdict(list)
    for r in q:
        if r["episode_id"] in tau_ep and sum(v < 0 for v in seq[r["machine_id"]]) < 0.1 * len(seq[r["machine_id"]]): eps[r["episode_id"]].append(r)
    _S.update(gm=gm, seq=seq, eps=eps)
    us = sorted({r["u"] for r in P}); jobs = []
    for e, ms in eps.items():
        L = len(seq[ms[0]["machine_id"]])
        jobs += [(e, u, int(round(tau_ep[e]))) for u in us if u < L and all(r["u"] != u or True for r in [])]
    jobs = [j for j in jobs if True]
    with get_context("fork").Pool(8) as pool: res = pool.map(job, jobs, chunksize=6)
    rows = [x for o in res for x in o]; y = {(r["mid"], r["u"]): (r["y"], r["p5"], r["p"]) for r in P}
    keep = []
    for x in rows:
        k = (x["mid"], x["u"])
        if k in y: x["y"], p5, p0 = y[k]; x["sz"] = np.array(p5[1:]) / max(p0, 1e-12); x["p_row"] = p0; keep.append(x)
    pickle.dump(keep, open("experiments/pooling/rows.pkl", "wb")); print(len(keep), "rows; max |p_prod - saved production p| =", max(abs(x["p_prod"] - x["p_row"]) for x in keep))

if __name__ == "__main__": main()
