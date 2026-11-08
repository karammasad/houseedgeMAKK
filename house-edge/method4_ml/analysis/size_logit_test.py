"""Offset-logistic correction of P(1x|win) (size model's logit fixed as offset) using whole-history payout counts.
Training examples from histories only (train/validation/test), evaluated on validation targets via common.evaluate.points."""
import random, numpy as np
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
from method4_ml.logit_stack import fit_offset_logistic, predict
PUB = "data/main"; tune, audit = folds(PUB)
m = Model().fit(PUB); base = m.predict(PUB, "validation")
Xq, _, ids = m._features(PUB, "validation", queries=True); SZ = m._sz[next(iter(m._sz))]
sz = m.size_models["gate"]; TY = {"steady": 0, "balanced": 1, "jackpot": 2}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def feats(r, s, a):
    w = [x for x in s if x > 0]; n = len(w); n1 = sum(x == 1 for x in w); r20 = w[-20:]
    f1 = (n1 + 1) / (n + 2); f20 = (sum(x == 1 for x in r20) + 1) / (len(r20) + 2)
    return a, n, f1, f20
def design(F, ty, a_off):
    n = F[:, 1]; c1 = (lg(F[:, 2]) - a_off) * n / (n + 20); c2 = (lg(F[:, 3]) - a_off) * len(F) * 0 + (lg(F[:, 3]) - a_off) * np.minimum(n, 20) / 20
    cols = []
    for t in range(3):
        msk = (ty == t); cols += [c1 * msk, c2 * msk, msk.astype(float)]
    return np.column_stack(cols)
rows = []
rng = random.Random(7)
for sp in ("train", "validation", "test"):
    q, seq = load_split(PUB, sp)
    for r in q:
        s = seq[r["machine_id"]]; T = [t for t in range(100, r["cutoff"]) if s[t] > 0]
        if len(T) < 4: continue
        for t in rng.sample(T, min(10, len(T))):
            rows.append((r, s[:t], s[t] == 1))
            if min(s) >= 0:
                keep = set(range(t - 40, t + 1)) | set(rng.sample(range(t - 40), 24))
                rows.append((r, [x if i in keep else -1 for i, x in enumerate(s[:t])], s[t] == 1))
print("examples", len(rows), flush=True)
F = []; y = []; ty = []
for r, s, lab in rows:
    F.append(feats(r, s, sz.predict(r, s)[0])); y.append(int(lab)); ty.append(TY[r["machine_type"]])
F = np.array(F); y = np.array(y); ty = np.array(ty); off = lg(F[:, 0])
q, seq = load_split(PUB, "validation"); R = {r["query_id"]: r for r in q}
Fq = np.array([feats(R[k], seq[R[k]["machine_id"]], SZ[k][0]) for k in ids]); tyq = np.array([TY[R[k]["machine_type"]] for k in ids])
pw = np.array([1 - base[k][0] for k in ids])
def score(a_new):
    out = {}
    for k, p, a in zip(ids, pw, a_new):
        sz0 = SZ[k]; a = min(max(a, 1e-4), 1 - 1e-4); rest = sum(sz0[1:])
        s = [a] + [x * (1 - a) / rest for x in sz0[1:]]
        out[k] = [1 - p] + [p * x for x in s]
    return points(PUB, out, tune)[0], points(PUB, out, audit)[0], points(PUB, out, None)[0]
b0 = score(Fq[:, 0]); print("baseline sizes      %.3f %.3f %.3f" % b0)
for l2 in (1, 10, 100, 1000):
    Z = design(F, ty, off); beta = fit_offset_logistic(Z, y, off, l2)
    an = predict(design(Fq, tyq, lg(Fq[:, 0])), beta, lg(Fq[:, 0])); s_ = score(an)
    print(f"l2={l2:5g} beta {np.round(beta, 2)}  {s_[0]:.3f} {s_[1]:.3f} {s_[2]:.3f}  d_full {s_[2]-b0[2]:+.3f}", flush=True)
