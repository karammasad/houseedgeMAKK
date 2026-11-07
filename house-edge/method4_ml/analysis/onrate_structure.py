"""Is the true ON-cell win rate a (near-)fixed function of (type, signal, size state)? (diagnosis: validation truth)
Size state = which of the type's two P(1x|win) levels; on-cell = true P(win) above a per-type threshold."""
import csv, statistics as st
from collections import defaultdict
from common.data import load_split
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
T = {r["query_id"]: [float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
q, _ = load_split(PUB, "validation")
lv = defaultdict(set)
for r in q:
    t = T[r["query_id"]]; w = 1 - t[0]; lv[r["machine_type"]].add(round(t[1] / w, 4))
lv = {k: sorted(v) for k, v in lv.items()}            # [low 1x share, high 1x share]
THR = {"steady": .5, "balanced": .5, "jackpot": .15}
cells = defaultdict(list)
for r in q:
    t = T[r["query_id"]]; w = 1 - t[0]
    if w < THR[r["machine_type"]]: continue
    state = int(round(t[1] / w, 4) == lv[r["machine_type"]][1])          # 1 = high 1x share
    cells[(r["machine_type"], state, r["signal"])].append(w)
tot = defaultdict(list)
for k in sorted(cells):
    v = cells[k]; tot[k[:2]] += v
    print(f"{k[0]:9s} 1x-state {k[1]} signal {k[2]:+d}  n={len(v):3d}  on-rate mean {st.mean(v):.3f}  sd {st.pstdev(v):.3f}")
print("\nby (type, size state) only:")
for k in sorted(tot): print(f"{k[0]:9s} 1x-state {k[1]}  n={len(tot[k]):3d}  mean {st.mean(tot[k]):.3f}  sd {st.pstdev(tot[k]):.3f}")
allw = defaultdict(list)
for k, v in cells.items(): allw[k[0]] += v
print("\nby type only:", {k: (round(st.mean(v), 3), round(st.pstdev(v), 3)) for k, v in allw.items()})
