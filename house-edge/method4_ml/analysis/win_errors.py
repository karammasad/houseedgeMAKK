"""Win-probability error structure of the v3 base gate (analysis only; validation targets used for diagnosis)."""
import csv
from collections import defaultdict
from common.data import load_split
from method4_ml.model import Model
from method4_ml.features3 import gate_summary
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
T = {r["query_id"]: 1 - float(r["p_0"]) for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
m = Model(blend=0).fit(PUB); q, seq = load_split(PUB, "validation")
rec = []
for r in q:
    g = gate_summary(seq[r["machine_id"]][:r["cutoff"]], r, m.gm)
    rec.append((r, g, T[r["query_id"]], C[r["query_id"]]))
def show(title, keyf):
    d = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for r, g, t, c in rec:
        k = keyf(r, g, t, c); e = (g[0] - t) ** 2; x = d[k]; x[0] += 1; x[1] += e; x[2] += g[0]; x[3] += t
    tot = sum(v[1] for v in d.values())
    print(title)
    for k in sorted(d): v = d[k]; print(f"  {str(k):28s} n={v[0]:4d} share of win SE {v[1]/tot:5.1%} mse {v[1]/v[0]:.4f} mean pred {v[2]/v[0]:.3f} mean true {v[3]/v[0]:.3f}")
show("by true P(win) bin", lambda r, g, t, c: round(t * 5) / 5)
show("by condition", lambda r, g, t, c: c)
show("by gate flip posterior P(flip)", lambda r, g, t, c: round(g[2] * 4) / 4)
show("by top structure mass", lambda r, g, t, c: round(g[4] * 4) / 4)
show("by condition x P(no gate)>0.5", lambda r, g, t, c: (c, g[3] > .5))

print("\nTop 25 queries by squared win-prob error (model = adi gate):")
tot = sum((g[0] - t) ** 2 for _, g, t, _ in rec)
top = sorted(rec, key=lambda x: -(x[1][0] - x[2]) ** 2)[:25]
for r, g, t, c in top:
    s = seq[r["machine_id"]]; w = [v for v in s if v >= 0]
    print(f"{c[:6]:6s} {r['machine_type'][:4]} sig{r['signal']:+d} pred {g[0]:.2f} (noflip {g[1]:.2f}) true {t:.2f} flip {g[2]:.2f} mass {g[4]:.2f} cut {r['cutoff']} last10 {''.join(str(int(v>0)) for v in s[-10:] )}")
print("top 25 share of total win SE:", round(sum((g[0] - t) ** 2 for _, g, t, _ in top) / tot, 3), " top 100:", round(sum((g[0] - t) ** 2 for _, g, t, _ in sorted(rec, key=lambda x: -(x[1][0] - x[2]) ** 2)[:100]) / tot, 3))
