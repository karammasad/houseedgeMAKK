"""Exact size menu (Person D). Data finding (analysis/generator_probe.py): the true size distribution given a win takes exactly TWO
vectors per machine type, identical across every machine. The size model's predictions drift off the segment between them; snapping
each prediction onto a learned segment removes that error (analysis/size_menu_learn.py).

Learning uses HISTORY ticks only (allowed for every split): for random winning ticks t of each machine, the size model's prediction
from seq[:t] (ticks strictly before t) gives lambda_t, the position on the current segment; the observed payout at t (a history
tick, never a query label) is regressed as onehot(y_t) ~ lambda_t V1 + (1 - lambda_t) V2; alternate. Deterministic (fixed seeds)."""
import hashlib, os, random
from multiprocessing import get_context
import numpy as np
from common.data import load_split

CL = {1: 0, 2: 1, 5: 2, 20: 3}
TYPES = ("steady", "balanced", "jackpot")
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
                jobs.append((r, list(s[:t])))          # inputs: strictly earlier ticks only
                y = np.zeros(4); y[CL[s[t]]] = 1; Y.append(y); TY.append(r["machine_type"])
    return jobs, np.array(Y), np.array(TY)


def fit_menu(pub, splits, sz, k=8, workers=1, iters=6):
    jobs, Y, TY = _rows(pub, splits, k); _S["sz"] = sz; S = None
    if workers > 1:
        try:
            with get_context("fork").Pool(workers) as pool:
                S = pool.map(_job, jobs, chunksize=64)
        except (OSError, PermissionError, ImportError, ValueError, NotImplementedError):
            S = None
    if S is None:
        S = [_job(j) for j in jobs]
    S = np.array(S, float); menu = {}
    for ty in TYPES:
        St, Yt = S[TY == ty], Y[TY == ty]
        if len(St) < 50:
            continue
        V1, V2 = St[St[:, 0].argmax()].copy(), St[St[:, 0].argmin()].copy()
        for _ in range(iters):
            d = V1 - V2; lam = np.clip((St - V2) @ d / max(d @ d, 1e-12), 0, 1)
            B = np.linalg.lstsq(np.column_stack([lam, 1 - lam]), Yt, rcond=None)[0]
            V1, V2 = np.clip(B[0], 1e-4, None), np.clip(B[1], 1e-4, None); V1 /= V1.sum(); V2 /= V2.sum()
        menu[ty] = (V1.tolist(), V2.tolist())
    return menu


def snap(s, V):
    """Project a size vector (given a win) onto the segment V1-V2; returns a list summing to 1."""
    if V is None:
        return s
    V1, V2 = np.array(V[0]), np.array(V[1]); d = V1 - V2; s = np.asarray(s, float)
    lam = float(np.clip((s - V2) @ d / max(d @ d, 1e-12), 0, 1)); out = V2 + lam * d
    return (out / out.sum()).tolist()
