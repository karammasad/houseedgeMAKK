"""Split each machine's win sizes by the gate cell of (a) its MAP win structure, (b) a random other structure,
(c) every structure (best). Train, long histories."""
import math, random
from collections import Counter
from common.data import load_split, wins
from method1_gate.gate import machine_tables, structures
q, seq = load_split("data/main", "train")
ST = structures(8, 2); rng = random.Random(0)
def split_gain(s, w, st):
    g = {0: [0, 0], 1: [0, 0]}
    for t in range(8, len(s)):
        if s[t] > 0 and all(w[t - l] >= 0 for l in st):
            x = 0
            for l in st: x ^= w[t - l]
            g[x][int(s[t] == 1)] += 1
    def L(c):
        n = sum(c); return sum(v * math.log((v + .5) / (n + 1)) for v in c)
    tot = [g[0][0] + g[1][0], g[0][1] + g[1][1]]
    return L(g[0]) + L(g[1]) - L(tot), g
res = {"map": [], "random": [], "best_of_all": []}; ex = []
for r in q:
    s = seq[r["machine_id"]]; w = wins(s)
    if sum(v >= 0 for v in s) < 150: continue
    tb = machine_tables(s, r["cutoff"], [], ST, .5, .5); ll = [t[0][0] for t in tb[:-1]]
    mp = ST[max(range(len(ST)), key=lambda i: ll[i])]
    gm, g = split_gain(s, w, mp); res["map"].append(gm)
    res["random"].append(split_gain(s, w, rng.choice([x for x in ST if x != mp]))[0])
    res["best_of_all"].append(max(split_gain(s, w, x)[0] for x in ST))
    if len(ex) < 6: ex.append((r["machine_type"], mp, g))
for k, v in res.items(): print(f"{k:12s} mean 1x-split log-lik gain {sum(v) / len(v):.2f}  (n={len(v)})")
for e in ex: print(e)
