"""Train-only held-out comparison of Thanos structure-screening thresholds (his code unmodified, imported from the read-only scratch copy).
Same protocol as his cv.py / my ablation: fold i = hyperparameters from the other folds' TRAIN histories; held-out rows = train outcomes at c..c-3.
Identical fits: the continuous per-type parameters theta come from ONE fit at the 1% rule; the support is then widened (threshold 0.001) and only the
discrete priors (pi over structures, P(d,e)) and the flip prior are re-estimated with theta fixed (exactly the EM steps of egp.fit_hyper)."""
import copy, csv, os, pickle, sys, time
import numpy as np
SRC = "/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-kfts2v/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad/thanos_src/house-edge"
sys.path.insert(0, SRC); sys.path.insert(0, "/Users/mominhassan/houseedgeMAKK/house-edge")
from common.data import load_split, public_dir
from thanos_method import egp, cv
from thanos_method.model import Model


def widen(H1, tr, thr, em_iters=40):
    """Same theta as H1, support = screen(min_mass=thr), pi / Pde by EM, then the flip prior."""
    H = copy.deepcopy(H1); keep, P = egp.screen(tr, min_mass=thr); H.K = [(G, l) for G, l, _ in keep]
    stats = [egp.window_stats(m, H.K, egp.window_end(m)) for m in tr]
    R = np.stack([s[0] for s in stats]); C = np.stack([s[1] for s in stats]); E = np.stack([s[2] for s in stats]); sig = np.array([m.signal for m in tr]); ty = np.array([m.type for m in tr])
    H.lpi = np.log(np.array([p for _, _, p in keep]) / sum(p for _, _, p in keep)); types = sorted(set(ty))
    for _ in range(em_iters):
        rk = np.zeros(len(H.K)); rde = np.zeros((2, 2))
        for t in types:
            sel = ty == t; T = egp.machine_terms(H.theta[t], R[sel], C[sel], sig[sel], H.K, H.lPde, E[sel]) + H.lpi[None, :, None, None, None]
            Z = egp.lse(T.reshape(T.shape[0], -1), 1); post = np.exp(T - Z[:, None, None, None, None]); rk += post.sum((0, 2, 3, 4)); rde += post.sum((0, 1, 4))
        H.lpi = np.log((rk + 1e-2) / (rk.sum() + 1e-2 * len(rk))); H.lPde = np.log((rde + .5) / (rde.sum() + 2))
    egp.fit_flip_prior(tr, H, log=lambda *a: None); return H


def run(i):
    pub = public_dir("/Users/mominhassan/houseedgeMAKK/house-edge/data/main"); q, seq = load_split(pub, "train"); ms = egp.machines_from_split(q, seq)
    labels = {r["query_id"]: r["outcome"] for r in csv.DictReader(open(os.path.join(pub, "train", "labels.csv")))}
    cq = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "train", "conditions.csv")))}; ce = {r["episode_id"]: cq[r["query_id"]] for r in q}
    fe = set(cv.folds(q, ce)[i]); tr = [m for m in ms if m.ep not in fe]; te = [m for m in ms if m.ep in fe]; R = cv.rows_for(te, labels); egp.TICKSET = "own"; t0 = time.time()
    M = Model("train", verbose=False, tickset="own").fit(None, machines=tr); H1 = M.H; print(f"fold {i} base fit {time.time()-t0:.0f}s, K={len(H1.K)}", flush=True)
    out = {}; tm = {}
    for name, H in (("thr1%", H1),):
        t1 = time.time(); out[name] = np.array([egp.machine_posterior(mm, H)[1] for mm, _, _ in R]); tm[name] = (len(H.K), time.time() - t1)
    for name, thr in (("thr0.2%", 0.002), ("thr0.1%", 0.001)):
        t1 = time.time(); H = widen(H1, tr, thr); t2 = time.time(); out[name] = np.array([egp.machine_posterior(mm, H)[1] for mm, _, _ in R]); tm[name] = (len(H.K), time.time() - t2, t2 - t1)
        print(f"fold {i} {name} K={len(H.K)} {time.time()-t0:.0f}s", flush=True)
    meta = [dict(fold=i, qid=mm.qid, ep=mm.ep, type=mm.type, cond=ce[mm.ep], lag=j, y=y, short=mm.short) for mm, y, j in R]
    pickle.dump(dict(preds=out, meta=meta, timing=tm, Kbase=H1.K, lpi=H1.lpi.tolist()), open(f"thr_fold{i}.pkl", "wb")); print(f"fold {i} done {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__": run(int(sys.argv[1]))
