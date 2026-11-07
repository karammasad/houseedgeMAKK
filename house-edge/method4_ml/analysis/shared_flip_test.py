"""Gate-level test of the shared-flip mixture (sizes fixed; selection on tuning, audit as check)."""
import time
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
from method4_ml.shared_flip import fit_pi, episode_pw
PUB = "data/main"; tune, audit = folds(PUB)
m = Model(blend=0).fit(PUB); gm = m.gm; m.predict(PUB, "validation"); SZ = m._sz[next(iter(m._sz))]
tq, ts = load_split(PUB, "train"); t0 = time.time(); pi = fit_pi(gm, tq, ts); print(f"learned pi = {pi} on train ({time.time()-t0:.0f}s)")
q, seq = load_split(PUB, "validation"); eps = {}
for r in q: eps.setdefault(r["episode_id"], []).append(r)
def run(pi_):
    pw, psh = {}, {}
    for e, ms in eps.items():
        out, p, _ = episode_pw(gm, ms, seq, pi_); pw.update(out)
        for r in ms: psh[r["query_id"]] = p
    return pw, psh
def score(pw):
    o = {k: [1 - p] + [p * x for x in SZ[k]] for k, p in pw.items()}
    return points(PUB, o, tune)[0], points(PUB, o, audit)[0], points(PUB, o, None)
import csv
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{PUB}/validation/conditions.csv"))}
base, _ = run(0.0); a = score(base)
print(f"independent (current gate): tuning {a[0]:.3f} audit {a[1]:.3f} full {a[2][0]:.3f}  ratios", {c: round(float(v), 4) for c, v in a[2][1].items()})
for pi_ in (pi, 0.3, 0.5):
    pw, psh = run(pi_); b = score(pw)
    print(f"shared mixture pi={pi_}: tuning {b[0]:.3f} audit {b[1]:.3f} full {b[2][0]:.3f}  ratios", {c: round(float(v), 4) for c, v in b[2][1].items()})
    for c in ("stable", "short_history", "drift", "shared_change"):
        v = [psh[k] for k in psh if C[k] == c]; print(f"      mean P(H_shared) {c:14s} {sum(v)/len(v):.3f}")
