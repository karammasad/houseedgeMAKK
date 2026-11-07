"""Gate + sizes with a per-machine random shift vs plain v5 gate (tuning + audit halves; sizes fixed). Stage 1 falsification."""
import copy, time
from common.data import load_split
from common.evaluate import points, folds
from method4_ml.model import Model
from method4_ml.pooled import pooled_split
from method4_ml.machine_shift import fit_machine_shift, win_prob
PUB = "data/main"; tune, audit = folds(PUB)
m = Model(blend=0).fit(PUB); base = m.gm; sz = m.size_models["gate"]
q, seq = load_split(PUB, "validation")
SZ = {r["query_id"]: sz.predict(r, seq[r["machine_id"]]) for r in q}
def score(pw):
    out = {r["query_id"]: [1 - pw[r["query_id"]]] + [pw[r["query_id"]] * x for x in SZ[r["query_id"]]] for r in q}
    return points(PUB, out, tune)[0], points(PUB, out, audit)[0], points(PUB, out, None)
p0 = base.win_probs(q, seq); a = score(p0)
print(f"plain v5 gate:   tuning {a[0]:.3f} audit {a[1]:.3f} full {a[2][0]:.3f}", flush=True)
t0 = time.time(); gm = copy.deepcopy(base); ch = fit_machine_shift(gm, *pooled_split(PUB, ("train",)))
print("chosen (mode, sigma, evidence gain):", {t: (x[0], x[1], round(x[2], 1)) for t, x in ch.items()}, f"fit {time.time() - t0:.0f}s", flush=True)
p1 = {r["query_id"]: win_prob(gm, r, seq[r["machine_id"]]) for r in q}; b = score(p1)
print(f"machine shift:   tuning {b[0]:.3f} audit {b[1]:.3f} full {b[2][0]:.3f}", {c: round(float(v), 4) for c, v in b[2][1].items()})
