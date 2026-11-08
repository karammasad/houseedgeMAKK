import sys, pickle, time, numpy as np
SRC = "/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-kfts2v/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad/thanos_src/house-edge"
sys.path.insert(0, SRC); sys.path.insert(0, "/Users/mominhassan/houseedgeMAKK/house-edge")
from common.data import load_split, public_dir
from thanos_method import egp
pub = public_dir("/Users/mominhassan/houseedgeMAKK/house-edge/data/main"); egp.TICKSET = "own"
ms = egp.machines_from_split(*load_split(pub, "train")); t0 = time.time(); keep, P = egp.screen(ms, min_mass=0.0); print("screen on", len(ms), "train machines:", round(time.time() - t0, 1), "s")
cand = [(egp.ALL_GATES[a], b + 1, float(P[a, b])) for a in range(P.shape[0]) for b in range(P.shape[1])]; cand.sort(key=lambda x: -x[2])
tot = sum(c[2] for c in cand); print("total mass", round(tot, 4), "| candidates", len(cand))
for thr in (0.01, 0.005, 0.003, 0.002, 0.001, 0.0005):
    k = [c for c in cand if c[2] >= thr]; print(f"threshold {thr:<7} keeps {len(k):3d} structures, retained mass {sum(c[2] for c in k):.4f}, discarded mass {1-sum(c[2] for c in k):.4f}")
print("\nrank  gate / size-lag   mass     (kept at 1% marked *)")
for i, (G, l, p) in enumerate(cand[:26]): print(f"{i+1:3d}   {str(G):10s} / {l}   {p:.4f} {'*' if p>=0.01 else ''}")
print("\nlargest DISCARDED (<1%) entries, grouped by gate:")
disc = [c for c in cand if c[2] < 0.01]; byg = {}
for G, l, p in disc: byg[G] = byg.get(G, 0) + p
for G, p in sorted(byg.items(), key=lambda x: -x[1])[:10]: print(f"   gate {str(G):8s} total discarded mass {p:.4f}")
true3 = {(1, 3): 1, (2, 5): 2, (4, 7): 4}
wrong_size = sum(p for G, l, p in cand if G in true3 and l != true3[G]); other_pairs = sum(p for G, l, p in cand if len(G) == 2 and G not in true3); singles = sum(p for G, l, p in cand if len(G) == 1); none = sum(p for G, l, p in cand if len(G) == 0)
print(f"\nmass by kind: true 3 pairs with their size lag {sum(p for G,l,p in cand if G in true3 and l==true3[G]):.3f} | true pairs, wrong size lag {wrong_size:.3f} | other pairs {other_pairs:.3f} | single-lag gates {singles:.3f} | no gate {none:.3f}")
pickle.dump((cand, P), open("screen_cand.pkl", "wb"))
