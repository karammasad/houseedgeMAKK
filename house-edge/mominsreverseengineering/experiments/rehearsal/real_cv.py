"""Test B: Thanos' own 4-fold episode-grouped CV protocol on real Round 1 TRAIN (his cv.folds / cv.rows_for), T0 vs T1 (no-gate adaptation). python3 real_cv.py <fold>"""
import os, sys, csv, time, pickle, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[v] = "2"
WT = "/private/tmp/claude-501/-Users-mominhassan-houseedgeMAKK/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad/thanos_wt/house-edge"
sys.path.insert(0, WT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common.data import load_split, public_dir
from thanos import egp, cv
from thanos.model import Model
import nogate_fix
i = int(sys.argv[1]); pub = public_dir("/Users/mominhassan/houseedgeMAKK/house-edge/data/main"); q, seq = load_split(pub, "train"); ms = egp.machines_from_split(q, seq)
labels = {r["query_id"]: r["outcome"] for r in csv.DictReader(open(os.path.join(pub, "train", "labels.csv")))}
cq = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "train", "conditions.csv")))}; ce = {r["episode_id"]: cq[r["query_id"]] for r in q}
fe = set(cv.folds(q, ce)[i]); tr = [m for m in ms if m.ep not in fe]; te = [m for m in ms if m.ep in fe]; R = cv.rows_for(te, labels); t0 = time.time(); out = {}; info = {}
orig_screen = egp.screen
for name, fix in (("T0", False), ("T1", True)):
    egp.screen = orig_screen
    if fix: nogate_fix.apply(egp)
    egp.TICKSET = "own"; M = Model("train", verbose=False, tickset="own").fit(None, machines=tr); H = M.H
    out[name] = np.array([egp.machine_posterior(mm, H)[1] for mm, _, _ in R])
    if fix:
        idx = [k for k, (G, l) in enumerate(H.K) if len(G) == 0]; info["nogate_mass"] = [float(math.exp(H.lpi[k])) for k in idx]
        info["mu_n_ls_n"] = {t: (round(float(egp.unpack(H.theta[t])[0]["mu_n"]), 2), round(float(math.exp(egp.unpack(H.theta[t])[0]["ls_n"])), 2)) for t in H.theta}
    info[name + "_K"] = len(H.K); print(f"fold {i} {name} done {time.time()-t0:.0f}s", flush=True)
meta = [dict(fold=i, qid=mm.qid, ep=mm.ep, type=mm.type, cond=ce[mm.ep], lag=j, y=y, short=mm.short) for mm, y, j in R]
pickle.dump(dict(preds=out, meta=meta, info=info), open(f"real_cv_fold{i}.pkl", "wb")); print("done", info, flush=True)
