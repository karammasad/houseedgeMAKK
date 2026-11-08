"""Learn the exact 2-vector size menu per type from HISTORIES only (train/validation/test; no targets), then snap v7's size
predictions onto the learned segment. Winning history ticks t: size model prediction s_t (ticks < t only) -> lambda_t = position
on the current segment; observed payout one-hot y_t ~ lambda_t V1 + (1 - lambda_t) V2 (least squares); alternate a few times.
Scored on validation truth (diagnosis); V's never see targets."""
import csv, random, numpy as np
from collections import Counter
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
PUB = "data/main"; tune, audit = folds(PUB); CL = {1: 0, 2: 1, 5: 2, 20: 3}
m = Model().fit(PUB); pred = m.predict(PUB, "validation"); sz = m.size_models["gate"]
rng = random.Random(11); S, Y, TY = [], [], []
for sp in ("train", "validation", "test"):
    q, seq = load_split(PUB, sp)
    for r in q:
        s = seq[r["machine_id"]]; Tk = [t for t in range(40, r["cutoff"]) if s[t] > 0]
        for t in rng.sample(Tk, min(8, len(Tk))):
            S.append(sz.predict(r, s[:t])); y = np.zeros(4); y[CL[s[t]]] = 1; Y.append(y); TY.append(r["machine_type"])
S, Y, TY = np.array(S), np.array(Y), np.array(TY); print("winning ticks", len(S), flush=True)
def fit_menu(S, Y, iters=6):
    i, j = S[:, 0].argmax(), S[:, 0].argmin(); V1, V2 = S[i].copy(), S[j].copy()
    for _ in range(iters):
        d = V1 - V2; lam = np.clip((S - V2) @ d / (d @ d), 0, 1)
        A = np.column_stack([lam, 1 - lam]); B = np.linalg.lstsq(A, Y, rcond=None)[0]
        V1, V2 = np.clip(B[0], 1e-4, None), np.clip(B[1], 1e-4, None); V1 /= V1.sum(); V2 /= V2.sum()
    return V1, V2
Vl = {ty: fit_menu(S[TY == ty], Y[TY == ty]) for ty in ("steady", "balanced", "jackpot")}
T = {r["query_id"]: np.array([float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]) for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
Q = {r["query_id"]: r for r in csv.DictReader(open(f"{PUB}/validation/queries.csv"))}
for ty in Vl:
    c = Counter(tuple(np.round(T[k][1:] / (1 - T[k][0]), 4)) for k in T if Q[k]["machine_type"] == ty)
    print(f"  {ty}: learned {np.round(Vl[ty][0], 3)} {np.round(Vl[ty][1], 3)} | true {[v for v, _ in c.most_common(2)]}")
def snap(P, Vd, alpha=1.0):
    out = {}
    for k, p in P.items():
        p = np.array(p); s = p[1:] / (1 - p[0]); V1, V2 = Vd[Q[k]["machine_type"]]; d = V1 - V2
        lam = np.clip((s - V2) @ d / (d @ d), 0, 1); sp = alpha * (V2 + lam * d) + (1 - alpha) * s
        out[k] = [p[0]] + list((1 - p[0]) * sp)
    return out
print(f"{'v7':28s} {points(PUB, pred, tune)[0]:.3f} {points(PUB, pred, audit)[0]:.3f} {points(PUB, pred, None)[0]:.3f}")
for a in (0.5, 1.0):
    P = snap(pred, Vl, a); print(f"{'snap learned V, a=%.1f' % a:28s} {points(PUB, P, tune)[0]:.3f} {points(PUB, P, audit)[0]:.3f} {points(PUB, P, None)[0]:.3f}")
