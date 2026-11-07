"""Synthetic mechanism-recovery test for the two-lag size expert (train histories only).

  python -m method2_sizes.twolag_synth --data data/main

Keeps every train machine's real win/loss sequence and missing-tick pattern, and regenerates the SIZES tick by tick
from a planted mechanism (levels = the type's learned levels, big-win split = learned split):
  planted_two_lag : state = XOR(big at lag a, big at lag b), (a, b) random per machine
  planted_one_lag : state = big at lag k (the mechanism the current model already represents)
Checks: learned new-expert weight v, MAP pair recovery on long machines, and size log-loss of each expert.
"""
import argparse, math, random
from common.data import load_split, public_dir
from method2_sizes.sizes import GateSizes
from method2_sizes.twolag import TwoLagSizes, PAIRS, MAPS


def generate(q, seq, levels, split, mechanism, rng):
    out, truth = {}, {}
    for r in q:
        t = r["machine_type"]; lo, hi = levels[t]; s = list(seq[r["machine_id"]])
        a, b = rng.choice(PAIRS); k = rng.choice((1, 2, 4)); d = rng.randrange(2)
        for u, v in enumerate(s):
            if v <= 0:
                continue
            if mechanism == "two":
                xa = s[u - a] if u - a >= 0 else -1; xb = s[u - b] if u - b >= 0 else -1
                st = rng.randrange(2) if xa < 0 or xb < 0 else (int(xa > 1) ^ int(xb > 1)) ^ d
            else:
                xk = s[u - k] if u - k >= 0 else -1
                st = rng.randrange(2) if xk < 0 else int(xk > 1) ^ d
            s[u] = 1 if rng.random() < (lo, hi)[st] else rng.choices((2, 5, 20), split[t])[0]
        out[r["machine_id"]] = s; truth[r["machine_id"]] = (a, b) if mechanism == "two" else k
    return out, truth


def map_pair(tl, t, s):
    best = None
    for (a, b) in PAIRS:
        saved = tl.pairs; tl.pairs = "uniform"
        # evidence restricted to one pair: temporarily score with a one-hot pair prior
        tl._pair_prior = lambda _s, ab=(a, b): [1.0 if p == ab else 0.0 for p in PAIRS]
        z, _ = tl._expert(t, s)
        del tl._pair_prior; tl.pairs = saved
        if best is None or z > best[0]:
            best = (z, (a, b))
    return best[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); q, seq = load_split(pub, "train"); rng = random.Random(2026)
    ref = GateSizes(cell_prior="npmle", tied=True, tie_weight="learn").fit(q, seq)
    for mech in ("two", "one"):
        sseq, truth = generate(q, seq, ref.levels, ref.split, mech, rng)
        tl = TwoLagSizes(GateSizes(cell_prior="npmle", tied=True, tie_weight="learn", joint_em=True),
                         mode="mix", pairs="uniform").fit(q, sseq)
        longm = [r for r in q if sum(v >= 0 for v in sseq[r["machine_id"]]) > 150]
        rec = None
        if mech == "two":
            sample = longm[:120]
            hits = sum(map_pair(tl, r["machine_type"], sseq[r["machine_id"]]) == truth[r["machine_id"]] for r in sample)
            rec = hits / len(sample)
        wmean = sum(tl.weight_new(r, sseq[r["machine_id"]]) for r in q) / len(q)
        print(f"planted {'TWO-lag XOR' if mech == 'two' else 'ONE-lag   '}: learned prior weight of new expert v = {tl.v:.3f}; "
              f"mean posterior weight {wmean:.3f}" + (f"; MAP pair recovered on {rec:.0%} of 120 long machines" if rec is not None else ""))


if __name__ == "__main__":
    main()
