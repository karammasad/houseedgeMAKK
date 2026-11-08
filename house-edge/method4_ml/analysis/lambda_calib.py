"""Recalibrate the size-state position lambda (which of the 2 menu vectors) on HISTORY ticks only, then score on validation.
Rows: winning history ticks t of every machine, full-history inputs AND short-masked copies (40 recent + 24 older ticks),
size model on ticks < t only, label = history payout at t. Model: lambda' = sigmoid(a_ty * logit(lambda) + b_ty + short terms),
max likelihood of the payout under lambda' V1 + (1 - lambda') V2."""
import csv, hashlib, random, numpy as np
from multiprocessing import get_context
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
import method4_ml.size_menu as SM
PUB = "data/main"; tune, audit = folds(PUB)
m = Model().fit(PUB); P = m.predict(PUB, "validation"); sz = m.size_models["gate"]; MENU = m.menu
jobs, Y, TY, SH = [], [], [], []
for sp in ("train", "validation", "test"):
    q, seq = load_split(PUB, sp)
    for r in q:
        s = seq[r["machine_id"]]; cand = [t for t in range(70, r["cutoff"]) if s[t] > 0]
        rng = random.Random(int(hashlib.sha256(("lc|" + r["machine_id"]).encode()).hexdigest()[:12], 16))
        for t in sorted(rng.sample(cand, min(16, len(cand)))):
            y = np.zeros(4); y[SM.CL[s[t]]] = 1
            jobs.append((r, list(s[:t]))); Y.append(y); TY.append(r["machine_type"]); SH.append(0)
            if min(s) >= 0:
                keep = set(range(t - 40, t)) | set(rng.sample(range(t - 40), 24))
                jobs.append((r, [x if i in keep else -1 for i, x in enumerate(s[:t])])); Y.append(y); TY.append(r["machine_type"]); SH.append(1)
SM._S["sz"] = sz
with get_context("fork").Pool(8) as pool: S = np.array(pool.map(SM._job, jobs, chunksize=64))
Y, TY, SH = np.array(Y), np.array(TY), np.array(SH); print("rows", len(S), "short", SH.mean(), flush=True)
lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
def lam_of(Sr, V1, V2):
    d = V1 - V2; return np.clip((Sr - V2) @ d / (d @ d), 0, 1)
def fit_cal(L, Yr, sh, V1, V2, l2=1.0):
    X = np.column_stack([lg(L), np.ones(len(L)), sh * lg(L), sh]); w = np.array([1.0, 0, 0, 0])
    pv1, pv2 = Yr @ V1, Yr @ V2     # likelihood of the observed payout under each vector
    for _ in range(50):
        z = X @ w; lam = 1 / (1 + np.exp(-z)); lik = lam * pv1 + (1 - lam) * pv2
        g = X.T @ (lam * (1 - lam) * (pv1 - pv2) / lik) - 2 * l2 * (w - [1, 0, 0, 0])
        # Gauss-Newton style step with numeric Hessian approximation
        r = lam * (1 - lam) * (pv1 - pv2) / lik; H = (X * (r ** 2)[:, None]).T @ X + 2 * l2 * np.eye(4)
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-7: break
    return w
Q = {r["query_id"]: r for r in csv.DictReader(open(f"{PUB}/validation/queries.csv"))}
qv, seqv = load_split(PUB, "validation"); NOBS = {r["query_id"]: sum(1 for x in seqv[r["machine_id"]] if x >= 0) for r in qv}
W = {}
for ty, (V1, V2) in MENU.items():
    V1, V2 = np.array(V1), np.array(V2); sel = TY == ty
    W[ty] = fit_cal(lam_of(S[sel], V1, V2), Y[sel], SH[sel].astype(float), V1, V2); print(ty, "a, b, a_short, b_short =", np.round(W[ty], 3))
out = {}
for k, p in P.items():
    p = np.array(p); ty = Q[k]["machine_type"]; V1, V2 = map(np.array, MENU[ty]); s = p[1:] / (1 - p[0]); L = lam_of(s[None], V1, V2)[0]
    sh = float(NOBS[k] <= 64); w = W[ty]; z = w[0] * lg(L) + w[1] + sh * (w[2] * lg(L) + w[3]); l2_ = 1 / (1 + np.exp(-z))
    out[k] = [p[0]] + list((1 - p[0]) * (V2 + l2_ * (V1 - V2)))
for nm, PP in (("v8", P), ("v8 + lambda recalibration", out)):
    print(f"{nm:28s} tuning {points(PUB, PP, tune)[0]:.3f}  audit {points(PUB, PP, audit)[0]:.3f}  full {points(PUB, PP, None)[0]:.3f}", points(PUB, PP, None)[1])
