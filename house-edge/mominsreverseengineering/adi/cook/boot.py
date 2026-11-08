"""Episode bootstrap on cached validation predictions: P(candidate beats baseline), per-condition deltas. Analysis only."""
import csv, json, random, sys
from collections import defaultdict
from common.evaluate import COLS, CONDITIONS
pub = "data/main"
def L(f):
    parts = f.split("+")
    Ps = [json.load(open(f"build/cook/{p}.json"))["validation"] for p in parts]
    return {k: [sum(P[k][i] for P in Ps) / len(Ps) for i in range(5)] for k in Ps[0]}
A, B = L(sys.argv[1]), L(sys.argv[2])
T = list(csv.DictReader(open(f"{pub}/validation/targets.csv")))
C = {r["query_id"]: r["condition"] for r in csv.DictReader(open(f"{pub}/validation/conditions.csv"))}
R = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(f"{pub}/validation/reference.csv"))}
ep = defaultdict(lambda: [[], [], []])
for t in T:
    q = t["query_id"]; tr = [float(t[c]) for c in COLS]
    e = lambda p: sum((a / sum(p) - b) ** 2 for a, b in zip(p, tr))
    d = ep[(C[q], t["episode_id"])]; d[0].append(e(A[q])); d[1].append(e(B[q])); d[2].append(e(R[q]))
byc = defaultdict(list)
for (c, e), (a, b, r) in ep.items():
    byc[c].append((sum(a) / len(a), sum(b) / len(b), sum(r) / len(r)))
def pts(sample):
    ra = rb = 0
    for c in CONDITIONS:
        xs = sample[c]; ref = sum(x[2] for x in xs)
        ra += sum(x[0] for x in xs) / ref; rb += sum(x[1] for x in xs) / ref
    return 60 / (1 + ra / 4), 60 / (1 + rb / 4)
a, b = pts(byc); print(f"{sys.argv[1]} {a:.3f} vs {sys.argv[2]} {b:.3f}  delta {a - b:+.3f}")
for c in CONDITIONS:
    xs = byc[c]; ref = sum(x[2] for x in xs)
    print(f"  {c:14s} ratio {sum(x[0] for x in xs)/ref:.4f} vs {sum(x[1] for x in xs)/ref:.4f}")
rng = random.Random(0); win = 0; N = 2000
for _ in range(N):
    s = {c: [rng.choice(byc[c]) for _ in byc[c]] for c in CONDITIONS}
    x, y = pts(s); win += x > y
print(f"  bootstrap P(better) = {win / N:.3f}")
