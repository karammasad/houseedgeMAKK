"""Cheap falsification: do Thanos's screening rules (frozen, imported unmodified from the read-only scratch copy of origin/main 360ebc6/8e00a50) keep the TRUE (gate, size lag) entries
when the Round 2 generator uses more lag pairs than main and/or a size lag that is not the smaller gate lag? Oracle simulator, 768 machines (576 long / 192 short), 5% no gate.
Scenarios: P pairs (3 = main-like, 6, 10) x size lag (tied = smaller lag | free = a fixed random lag per pair). Coverage = share of gated machines whose true (G, l) is in the kept
support; also the number of kept structures and the screen mass on true entries."""
import sys, random, numpy as np
SRC = "/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-kfts2v/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad/thanos_src/house-edge"
sys.path.insert(0, SRC); sys.path.insert(0, "/Users/mominhassan/houseedgeMAKK/house-edge"); sys.path.insert(0, "../oracle")
from thanos_method import egp
import sim, gen2
egp.TICKSET = "own"; size_p = sim.size_dist(); ALLP = [(i, j) for i in range(1, 9) for j in range(i + 1, 9)]


def world(seed, P, tied, n=768, f_none=0.05):
    rng = random.Random(seed); pairs = rng.sample(ALLP, P); lag = {pr: (min(pr) if tied else rng.randint(1, 8)) for pr in pairs}; pop = sim.population(); out = []
    for i in range(n):
        ty = sim.TYPES[i % 3]; sg = rng.choice((-1, 0, 1)); L = rng.randint(192, 256); P0 = pop[ty]
        if rng.random() < f_none: pr = None; pp = [0, 0, 0, 1.0]; use = [(1, 2, 1)] * 3
        else: pr = rng.choice(pairs); use = [(pr[0], pr[1], lag[pr])] * 3; pp = [1.0, 0, 0, 0]
        m = gen2.gen_machine(rng, {ty: dict(cp=P0["cp"], pairs=use, pair_p=pp)}, ty, sg, L, size_p)
        keep = None
        if i % 4 == 3: keep = set(range(L - 40, L)) | set(rng.sample(range(0, L - 40), 24))
        y = [v if keep is None or t in keep else -1 for t, v in enumerate(m["pay"])]
        out.append((egp.Machine(f"q{i}", f"e{i//8}", f"m{i}", ty, sg, y), (pr, lag.get(pr)) if pr else None))
    return out


def run(seed, P, tied):
    W = world(seed, P, tied); ms = [m for m, _ in W]; keep, Pm = egp.screen(ms, min_mass=0.0); cand = {(G, l): Pm[egp.ALL_GATES.index(G), l - 1] for G in egp.ALL_GATES for l in range(1, 9)}
    truth = [t for _, t in W if t]; res = {}
    for name, thr in (("1%", 0.01), ("0.2%", 0.002)):
        K = {c for c, p in cand.items() if p >= thr}; cov = np.mean([(tuple(pr), l) in K for pr, l in truth]); res[name] = (len(K), cov, sum(cand[c] for c in K))
    pair_mass = {}
    for (G, l), p in cand.items(): pair_mass[G] = pair_mass.get(G, 0) + p
    pairs = {tuple(pr) for pr, _ in truth}; res["pair_mass_on_true_pairs"] = sum(pair_mass[p] for p in pairs); res["pair_cov_1%"] = np.mean([pair_mass[tuple(pr)] >= 0.01 for pr, _ in truth])
    return res


if __name__ == "__main__":
    print("coverage = share of gated machines whose TRUE (gate pair, size lag) is in the kept support; K = structures kept; mean over 6 seeds")
    print(f"{'pairs':>5s} {'size lag':>9s} | {'K@1%':>5s} {'cov@1%':>7s} {'mass@1%':>8s} | {'K@0.2%':>6s} {'cov@0.2%':>9s} | {'mass on true pairs':>19s} {'pair-level cov@1%':>18s}")
    for P in (3, 6, 10):
        for tied in (True, False):
            r = [run(s, P, tied) for s in range(6)]; g = lambda k, i: np.mean([x[k][i] for x in r])
            print(f"{P:5d} {'tied' if tied else 'free':>9s} | {g('1%',0):5.1f} {g('1%',1):7.3f} {g('1%',2):8.3f} | {g('0.2%',0):6.1f} {g('0.2%',1):9.3f} | {np.mean([x['pair_mass_on_true_pairs'] for x in r]):19.3f} {np.mean([x['pair_cov_1%'] for x in r]):18.3f}")
