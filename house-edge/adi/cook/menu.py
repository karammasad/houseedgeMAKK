"""Exact two-vector size menu, standard library only (adi port of method4_ml/size_menu.py, Kareem's v8 finding).

Finding (method4 RESEARCH_LOG #34-35): given a win, the true size distribution [1x, 2x, 5x, 20x] is one of exactly TWO vectors
per machine type, identical across machines. Size models drift off the segment between them; snapping each prediction onto
a learned segment removes that error.

Learning uses HISTORY ticks only (allowed for every split): for up to k random winning ticks t per machine, the size model's
prediction from seq[:t] (strictly earlier ticks) gives lambda_t, its position on the current segment; the observed payout at t
(a history tick, never a query label) is regressed as onehot(y_t) ~ lambda_t V1 + (1 - lambda_t) V2 by least squares;
alternate. Same algorithm and seeds as method4_ml/size_menu.py, with numpy replaced by plain Python.
"""
import hashlib, os, random
from multiprocessing import get_context
from common.data import load_split

CL = {1: 0, 2: 1, 5: 2, 20: 3}
_S = {}


def _job(args):
    r, s_trunc = args
    return _S["sz"].predict(r, s_trunc)


def _rows(pub, splits, k):
    jobs, Y, TY = [], [], []
    for sp in splits:
        if not os.path.exists(os.path.join(pub, sp, "history.csv")):
            continue
        q, seq = load_split(pub, sp)
        for r in q:
            s = seq[r["machine_id"]]; cand = [t for t in range(40, r["cutoff"]) if s[t] > 0]
            rng = random.Random(int(hashlib.sha256(("menu|" + r["machine_id"]).encode()).hexdigest()[:12], 16))
            for t in sorted(rng.sample(cand, min(k, len(cand)))):
                jobs.append((r, list(s[:t])))            # inputs: strictly earlier ticks only
                y = [0.0] * 4; y[CL[s[t]]] = 1.0; Y.append(y); TY.append(r["machine_type"])
    return jobs, Y, TY


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _lam(s, V1, V2):
    d = [a - b for a, b in zip(V1, V2)]; dd = max(_dot(d, d), 1e-12)
    return min(max(_dot([a - b for a, b in zip(s, V2)], d) / dd, 0.0), 1.0)


def _segment(St, Yt, iters):
    i1 = max(range(len(St)), key=lambda i: St[i][0]); i2 = min(range(len(St)), key=lambda i: St[i][0])
    V1, V2 = list(St[i1]), list(St[i2])
    for _ in range(iters):
        lam = [_lam(s, V1, V2) for s in St]
        # least squares Y ~ [lam, 1 - lam] @ [V1; V2]: 2x2 normal equations, one right-hand side per size class
        aa = sum(l * l for l in lam); ab = sum(l * (1 - l) for l in lam); bb = sum((1 - l) ** 2 for l in lam)
        det = aa * bb - ab * ab
        if abs(det) < 1e-12:
            break
        nv1, nv2 = [], []
        for j in range(4):
            ay = sum(l * y[j] for l, y in zip(lam, Yt)); by = sum((1 - l) * y[j] for l, y in zip(lam, Yt))
            nv1.append((bb * ay - ab * by) / det); nv2.append((aa * by - ab * ay) / det)
        V1 = [max(x, 1e-4) for x in nv1]; V2 = [max(x, 1e-4) for x in nv2]
        z1, z2 = sum(V1), sum(V2); V1 = [x / z1 for x in V1]; V2 = [x / z2 for x in V2]
    return V1, V2


def fit_menu(pub, splits, sz, k=32, workers=1, iters=6):
    """{machine_type: (V1, V2)} learned from history ticks of `splits` with size model `sz` (sz.predict(r, seq) -> 4 probs)."""
    jobs, Y, TY = _rows(pub, splits, k); _S["sz"] = sz
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            S = pool.map(_job, jobs, chunksize=64)
    else:
        S = [_job(j) for j in jobs]
    menu = {}
    for ty in sorted(set(TY)):
        idx = [i for i, t in enumerate(TY) if t == ty]
        if len(idx) < 50:
            continue
        menu[ty] = _segment([list(map(float, S[i])) for i in idx], [Y[i] for i in idx], iters)
    return menu


def snap(s, V):
    """Project a size vector (given a win) onto the segment V1-V2; returns a list summing to 1."""
    if V is None:
        return list(s)
    V1, V2 = V; lam = _lam(s, V1, V2)
    out = [b + lam * (a - b) for a, b in zip(V1, V2)]; z = sum(out)
    return [x / z for x in out]
