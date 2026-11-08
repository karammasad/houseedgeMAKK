"""Train-only ablation of Kareem's EGP using his code UNMODIFIED (read-only copy exported to the scratchpad; this harness only imports it).
Protocol = his cv.py: 4 episode-grouped folds (same fold function), hyperparameters from the other 3 folds' TRAIN histories, held-out fold scored
on train outcomes at cutoffs c, c-1, c-2, c-3 (history truncated), 5-class Brier. No validation/audit data, no conditions used for fitting.
Variants (per fold):  full (tickset own)  |  nodir: direction tie removed (P(d,e) fixed independent during the fit)  |
                      flat-pos: learned P(no flip) but flat position prior  |  flat-flip: P(no flip)=.5 and flat position (like our fixed p_change .5, uniform tau)  |
                      common: pre-registered 'common' tick set (the E2 question: own vs common on short logs)."""
import copy, os, sys, pickle, time, math
import numpy as np
SRC = "/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-kfts2v/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad/thanos_src/house-edge"
sys.path.insert(0, SRC); sys.path.insert(0, "/Users/mominhassan/houseedgeMAKK/house-edge")
from common.data import load_split, public_dir
from thanos_method import egp, cv
from thanos_method.model import Model
import csv


def fit_hyper_fixed_pde(machines, types=None, outer=3, em_iters=40):
    """Line-for-line copy of egp.fit_hyper except lPde stays at the independent prior (log .25) and is never updated."""
    H = egp.Hyper(); types = types or sorted({m.type for m in machines}); keep, P = egp.screen(machines)
    H.K = [(G, l) for G, l, _ in keep]
    stats = [egp.window_stats(m, H.K, egp.window_end(m)) for m in machines]
    R = np.stack([s[0] for s in stats]); C = np.stack([s[1] for s in stats]); E = np.stack([s[2] for s in stats]); sig = np.array([m.signal for m in machines])
    ty = np.array([m.type for m in machines]); H.lpi = np.log(np.array([p for _, _, p in keep]) / sum(p for _, _, p in keep))
    x0 = {t: egp.starts([m for m in machines if m.type == t]) for t in types}
    for it_ in range(outer):
        for t in types:
            sel = ty == t; r = egp.fit_type(R[sel], C[sel], sig[sel], H.K, H.lpi, H.lPde, x0[t], E[sel]); H.theta[t] = r.x; x0[t] = [r.x]
        for _ in range(em_iters):
            rk = np.zeros(len(H.K))
            for t in types:
                sel = ty == t
                T = egp.machine_terms(H.theta[t], R[sel], C[sel], sig[sel], H.K, H.lPde, E[sel]) + H.lpi[None, :, None, None, None]
                Z = egp.lse(T.reshape(T.shape[0], -1), 1); post = np.exp(T - Z[:, None, None, None, None]); rk += post.sum((0, 2, 3, 4))
            H.lpi = np.log((rk + 1e-2) / (rk.sum() + 1e-2 * len(rk)))
    return H


def predict(rows, H, tickset):
    egp.TICKSET = tickset
    return np.array([egp.machine_posterior(mm, H)[1] for mm, _, _ in rows])


def run_fold(i):
    pub = public_dir("/Users/mominhassan/houseedgeMAKK/house-edge/data/main"); q, seq = load_split(pub, "train"); ms = egp.machines_from_split(q, seq)
    labels = {r["query_id"]: r["outcome"] for r in csv.DictReader(open(os.path.join(pub, "train", "labels.csv")))}
    cq = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "train", "conditions.csv")))}; ce = {r["episode_id"]: cq[r["query_id"]] for r in q}
    fe = set(cv.folds(q, ce)[i]); tr = [m for m in ms if m.ep not in fe]; te = [m for m in ms if m.ep in fe]; R = cv.rows_for(te, labels); t0 = time.time(); out = {}
    M = Model("train", verbose=False, tickset="own").fit(None, machines=tr); Hf = M.H; print(f"fold {i} full fit {time.time()-t0:.0f}s", flush=True)
    out["full"] = predict(R, Hf, "own")
    Hp = copy.deepcopy(Hf); Hp.hist = np.full(egp.NBINS, 1.0 / egp.NBINS); out["flat-pos"] = predict(R, Hp, "own")
    Hq = copy.deepcopy(Hp); Hq.rho = {False: .5, True: .5}; out["flat-flip"] = predict(R, Hq, "own"); print(f"fold {i} flip variants {time.time()-t0:.0f}s", flush=True)
    egp.TICKSET = "own"; Hn = fit_hyper_fixed_pde(tr); egp.fit_flip_prior(tr, Hn, log=lambda *a: None); out["nodir"] = predict(R, Hn, "own"); print(f"fold {i} nodir {time.time()-t0:.0f}s", flush=True)
    Mc = Model("train", verbose=False, tickset="common").fit(None, machines=tr); out["common"] = predict(R, Mc.H, "common"); print(f"fold {i} common {time.time()-t0:.0f}s", flush=True)
    meta = [dict(fold=i, qid=mm.qid, ep=mm.ep, type=mm.type, cond=ce[mm.ep], lag=j, y=y, short=mm.short) for mm, y, j in R]
    info = dict(Pde_full=np.exp(Hf.lPde).tolist(), rho_full=Hf.rho, hist_full=Hf.hist.tolist(), Pde_nodir=np.exp(Hn.lPde).tolist(), rho_nodir=Hn.rho, rho_common=Mc.H.rho)
    pickle.dump(dict(preds=out, meta=meta, info=info), open(f"fold{i}.pkl", "wb")); print(f"fold {i} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run_fold(int(sys.argv[1]))
