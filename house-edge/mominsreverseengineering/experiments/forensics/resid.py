"""Residual forensics for the v9 + shared-base gate + size model (cached fit, no refit). TRAIN ONLY: pseudo-queries inside long train logs
(target u, history = all ticks < u) and the 768 real train query outcomes (labels.csv; never read by fit). Machine-level descriptors
(flip time, top structure, gate strength) come from the production posterior on the machine's full history (stratification only)."""
import csv, math, pickle, sys, time
from collections import defaultdict
from multiprocessing import get_context
import numpy as np
from common.data import public_dir, load_split
from adi import gate as G
from method4_ml.size_menu import snap

SIZES = (1, 2, 5, 20); _S = {}


def post(gm, ss, cut, ty, sg):
    r = dict(machine_type=ty, signal=sg); cfg = gm._mcfg(ss, r_type=ty); taus, lp = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(ss, cut, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r), cfg["half_life"]); sp = gm._sp(ss)
    J, PR, TI, SI = [], [], [], []
    for si, rows in enumerate(tb):
        if sp[si] <= 0: continue
        for ti, (ll, pr) in enumerate(rows): J.append(ll + math.log(sp[si]) + lp[ti]); PR.append(pr); TI.append(ti); SI.append(si)
    J = np.array(J); w = np.exp(J - J.max()); w /= w.sum(); TI = np.array(TI); SI = np.array(SI); nt = len(taus); fl = TI < nt
    Pf = float(w[fl].sum()); mt = np.bincount(TI[fl], weights=w[fl], minlength=nt)
    tau = float((mt * np.array(taus)).sum() / Pf) if Pf > 1e-9 else None
    sm = np.bincount(SI, weights=w, minlength=len(gm.ST) + 1)
    return float((w * np.array(PR)).sum()), Pf, tau, sm


def strength(s, pair, tau):
    if pair is None: return None, None
    i, j = pair[:2]; n = np.zeros(2); k = np.zeros(2)
    for t in range(max(i, j), len(s)):
        if s[t] < 0 or s[t - i] < 0 or s[t - j] < 0: continue
        g = int(s[t - i] > 0) ^ int(s[t - j] > 0)
        if tau is not None and t >= tau: g ^= 1
        n[g] += 1; k[g] += s[t] > 0
    r = (k + .5) / (n + 1); return float(r.max()), float(r.min())


def job(args):
    r, = args; gm, g, menu = _S["gm"], _S["g"], _S["menu"]; s = _S["seq"][r["machine_id"]]; L = len(s); ty, sg = r["machine_type"], r["signal"]
    pw, Pf, tau, sm = post(gm, s, L, ty, sg); top = int(np.argmax(sm)); st = gm.ST[top] if top < len(gm.ST) else ("none", None)
    pair = st[1][:3] if st[1] is not None else None; flipped = Pf > .5
    son, soff = strength(s, pair, tau if flipped else None)
    mach = dict(mid=r["machine_id"], ep=r["episode_id"], type=ty, sg=sg, cond=_S["cond"][r["query_id"]], L=L, Pf=Pf, tau=tau if flipped else None,
                form=st[0], pair=tuple(pair[:2]) if pair else None, smass=float(sm[top]), son=son, soff=soff, short=sum(v < 0 for v in s) > L / 2)
    rows = []
    def row(u, y, kind):
        ss = list(s[:u]); p, pf_u, _, _ = post(gm, ss, u, ty, sg); sz = np.array(snap(g.predict(dict(r, cutoff=u), ss), menu.get(ty)))
        p5 = np.concatenate([[1 - p], p * sz]); last = [v for v in ss if v >= 0]
        k = pair[2] if pair and len(pair) > 2 else 1
        bigs = [t for t, v in enumerate(ss) if v >= 5]
        rows.append(dict(mach, kind=kind, u=u, y=y, p=p, p5=p5, Pf_u=pf_u, last=last[-1] if last else -1, lagk=ss[u - k] if u - k >= 0 else -1,
                         since_big=(u - bigs[-1]) if bigs else 999, nobs=len(last)))
    if not mach["short"]:
        for u in range(int(.30 * L), L, 4): row(u, s[u], "pseudo")
    row(L, _S["lab"][r["query_id"]], "label")
    return rows


def main():
    pub = public_dir("../data/main"); t0 = time.time()
    from method4_ml.model import Model
    m = Model(cache=True, shared_base=True).fit(pub); g = m.size_models["gate"]; g.win_dir_lam = 0.0
    q, seq = load_split(pub, "train")
    cond = {x["query_id"]: x["condition"] for x in csv.DictReader(open("../data/main/train/conditions.csv"))}
    lab = {x["query_id"]: int(x["outcome"]) for x in csv.DictReader(open("../data/main/train/labels.csv"))}
    _S.update(gm=m.gm, g=g, menu=m.menu, seq=seq, cond=cond, lab=lab)
    with get_context("fork").Pool(8) as p: out = p.map(job, [(r,) for r in q], chunksize=4)
    rows = [x for o in out for x in o]; pickle.dump(rows, open("experiments/forensics/rows.pkl", "wb"))
    print(len(rows), "rows", round(time.time() - t0), "s")

if __name__ == "__main__": main()
