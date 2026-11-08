"""Does the 1% screening rule keep a RARE new structure? Simulated train sets (oracle simulator, known truth) where a fourth lag pair (3,6) with size lag 3
appears in a fraction f of machines (a generator change), 768 machines per set (576 long / 192 short, as in train). Reports the screen's recovered mass for
the new pair and whether it survives min_mass = 1% vs 0.2%. Also a no-gate share test."""
import sys, random, numpy as np
SRC = "/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-kfts2v/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad/thanos_src/house-edge"
sys.path.insert(0, SRC); sys.path.insert(0, "/Users/mominhassan/houseedgeMAKK/house-edge"); sys.path.insert(0, "../oracle")
from thanos_method import egp
import sim
egp.TICKSET = "own"; size_p = sim.size_dist(); NEW = (3, 6)


def world(seed, f_new, f_none, n=768):
    rng = random.Random(seed); pop = sim.population(); out = []
    for t in sim.TYPES:
        base = list(pop[t]["pairs"]); w = np.array(pop[t]["pair_p"][:3]); w = w / w.sum() * (1 - f_new - f_none)
        pop[t]["pairs"] = base + [(3, 6, 3)]; pop[t]["pair_p"] = list(w) + [f_new, f_none]       # index 3 = new pair; index 4 = none
    for i in range(n):
        ty = sim.TYPES[i % 3]; sg = rng.choice((-1, 0, 1)); L = rng.randint(192, 256); P = pop[ty]
        P2 = dict(P); P2["pair_p"] = P["pair_p"]
        m = sim.gen_machine(rng, {ty: P2}, ty, sg, L, size_p) if False else None
        # generate with 5 outcomes: map index 3 -> new pair, 4 -> none (gen_machine treats index 3 as none, so patch pairs/probs accordingly)
        pairs = P["pairs"][:3] + [(3, 6, 3)]; probs = P["pair_p"][:3] + [P["pair_p"][3]] + [P["pair_p"][4]]
        k = rng.choices(range(5), probs)[0]
        Q = {ty: dict(cp=P["cp"], pairs=pairs[:3] + [None], pair_p=None)}
        # reuse gen_machine by forcing the structure: temporary population with a single structure chosen
        if k == 4: pp = [0, 0, 0, 1.0]; pr = pairs[:3]
        elif k == 3: pp = [1.0, 0, 0, 0]; pr = [(3, 6, 3), pairs[0], pairs[1]]
        else: pp = [1.0 if j == k else 0 for j in range(3)] + [0]; pr = pairs[:3]
        m = sim.gen_machine(rng, {ty: dict(cp=P["cp"], pairs=pr, pair_p=pp)}, ty, sg, L, size_p)
        short = (i % 4 == 3); keep = None
        if short: keep = set(range(L - 40, L)) | set(rng.sample(range(0, L - 40), 24))
        y = [v if keep is None or t in keep else -1 for t, v in enumerate(m["pay"])]
        out.append((egp.Machine(f"q{i}", f"e{i//8}", f"m{i}", ty, sg, y), k))
    return out


def run(f_new, f_none, seed):
    W = world(seed, f_new, f_none); ms = [m for m, _ in W]; keep, P = egp.screen(ms, min_mass=0.0)
    newm = sum(P[egp.ALL_GATES.index((3, 6)), l - 1] for l in range(1, 9)); newm_l3 = P[egp.ALL_GATES.index((3, 6)), 2]; nonem = sum(P[0, l] for l in range(8))
    return sum(1 for _, k in W if k == 3), newm, newm_l3, nonem, sum(1 for _, k in W if k == 4)


if __name__ == "__main__":
    print("A. new pair (3,6)/size lag 3 at true share f, 768 machines, 8 seeds each: screen mass for (3,6) any size lag | at lag 3 ; survives 1% / 0.2%")
    for f in (0.005, 0.01, 0.02, 0.03, 0.05):
        r = [run(f, 0.05, s) for s in range(8)]; m = np.array([x[1] for x in r]); m3 = np.array([x[2] for x in r])
        print(f"   f={f:.3f} (≈{np.mean([x[0] for x in r]):.1f} machines): mass any-lag {m.mean():.4f} [{m.min():.4f},{m.max():.4f}]  at true lag {m3.mean():.4f}  kept at 1%: {np.mean(m3>=0.01):.2f}  kept at 0.2%: {np.mean(m3>=0.002):.2f}")
    print("B. no-gate share (true 5% / 15% / 30%): screen mass on the no-gate candidate (any size lag)")
    for fn in (0.05, 0.15, 0.30):
        r = [run(0.0, fn, s) for s in range(4)]; print(f"   true none share {fn:.2f} (≈{np.mean([x[4] for x in r])/768:.3f} of machines): screen mass {np.mean([x[3] for x in r]):.4f}")
