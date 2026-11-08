"""Synthetic CHANGED-generator kit in the official file format, for testing the Round 2 adaptation procedure (adapt.py). NOT a claim about the real
Round 2: it only exercises the mechanics (new lag pairs, a different size-shift, optional absence of the size mechanism) with known truth.

  python3 experiments/synth_kit.py --out /path/synth --pairs 2,4:3,6:1,5 --gamma 0.5 [--episodes 10]
Layout written: <out>/public/{metadata.json, train|validation|test/{queries,history,...}.csv}.
Generator: machine type + signal; one gate pair from the menu (per machine); XOR parity of win bits at the two lags; on/off cells with logit rates
mu[type] + slope*signal + N(0, sigma^2); one hidden size state per tick = payout category (loss / 1x / big) at the smaller lag, direction random per machine,
which shifts the win logit by +-gamma and selects one of two size vectors per type; drift / shared_change conditions flip the polarity at tau ~ 0.68 * cutoff
(shared: one tau per episode); short_history logs keep 40 consecutive + 24 sampled ticks. Deterministic.
"""
import argparse, csv, json, math, os, random

TYPES = {"steady": dict(on=2.9, off=-1.8, V=([.455, .436, .107, .002], [.754, .197, .048, .001])),
         "balanced": dict(on=1.7, off=-3.0, V=([.5, .348, .13, .022], [.214, .547, .205, .034])),
         "jackpot": dict(on=0.4, off=-4.4, V=([.335, .245, .315, .105], [.121, .324, .416, .139]))}
PAYOUTS = (1, 2, 5, 20)
COND = ("stable", "short_history", "drift", "shared_change")
sig = lambda x: 1 / (1 + math.exp(-x))


def cat(v):
    return 0 if v == 0 else (1 if v == 1 else 2)


def simulate(rng, ty, signal, pair, cutoff, flip_at, gamma, slope, sigma, direction):
    p = TYPES[ty]; i, j = pair; k = min(pair)
    a_on = p["on"] + slope * signal + rng.gauss(0, sigma); a_off = p["off"] + slope * signal + rng.gauss(0, sigma)
    hist = []; maxlag = max(pair)

    def state(t):                    # +1 = big-size state
        c = cat(hist[t - k]); return 1 if ((c == 2) != (direction == 1)) else -1

    def probs(t):
        if t < maxlag:
            pw, sv = sig(p["on"] * .5 + p["off"] * .5), None
            return pw, p["V"][0]
        x = (1 if hist[t - i] > 0 else 0) ^ (1 if hist[t - j] > 0 else 0)
        if flip_at is not None and t >= flip_at:
            x ^= 1
        s = state(t); a = a_on if x == 1 else a_off
        return sig(a + gamma * s), p["V"][1 if s == 1 else 0]

    for t in range(cutoff):
        pw, V = probs(t)
        hist.append(0 if rng.random() > pw else rng.choices(PAYOUTS, weights=V)[0])
    pw, V = probs(cutoff)
    truth = [1 - pw] + [pw * v for v in V]
    out = 0 if rng.random() > pw else rng.choices(PAYOUTS, weights=V)[0]
    return hist, truth, out


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def build(out, menu, gamma, episodes, seed=11, slope=0.34, sigma=0.3, tag="synth"):
    rng = random.Random(seed); pub = os.path.join(out, "public")
    splits = {"train": episodes, "validation": max(4, episodes // 2), "test": max(4, episodes // 2)}
    pooled = {}
    for split, neps in splits.items():
        Q, Hh, Lb, Cn, Tg = [], [], [], [], []; e = 0
        for cond in COND:
            for _ in range(neps):
                eid = f"{tag}-{split}-e{e:05d}"; e += 1; cutoff = rng.randint(192, 256)
                tau_shared = int(cutoff * rng.uniform(.62, .76)) if cond == "shared_change" else None
                for m in range(8):
                    mid = f"{eid}-m{m:03d}"; ty = rng.choice(list(TYPES)); sg = rng.choice([-1, 0, 1]); pair = rng.choice(menu)
                    flip = None
                    if cond == "drift":
                        flip = int(cutoff * rng.uniform(.62, .76))
                    elif cond == "shared_change":
                        flip = tau_shared
                    hist, truth, outc = simulate(rng, ty, sg, pair, cutoff, flip, gamma, slope, sigma, rng.randint(0, 1))
                    qid = f"{mid}-next"; Q.append([qid, eid, mid, ty, sg, cutoff])
                    keep = range(cutoff)
                    if cond == "short_history":
                        keep = sorted(set(range(cutoff - 40, cutoff)) | set(rng.sample(range(0, cutoff - 40), 24)))
                    Hh += [[eid, mid, t, hist[t]] for t in keep]
                    Lb.append([qid, outc]); Cn.append([qid, cond]); Tg.append([qid, eid] + truth + [outc])
                    if split == "train":
                        for v in hist:
                            pooled.setdefault(ty, [0] * 5)[(0, 1, 2, 5, 20).index(v)] += 1
        d = os.path.join(pub, split)
        write(os.path.join(d, "queries.csv"), ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"], Q)
        write(os.path.join(d, "history.csv"), ["episode_id", "machine_id", "tick", "payout"], Hh)
        if split != "test":
            write(os.path.join(d, "labels.csv"), ["query_id", "outcome"], Lb); write(os.path.join(d, "conditions.csv"), ["query_id", "condition"], Cn)
        if split == "validation":
            write(os.path.join(d, "targets.csv"), ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"], Tg)
            ref = []
            for r in Q:
                c = pooled[r[3]]; s = sum(c); ref.append([r[0]] + [x / s for x in c])
            write(os.path.join(d, "reference.csv"), ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], ref)
    json.dump(dict(round="synthetic", max_points=60, season_id="0" * 24, suite_version="experimentation-suite-v3", note="synthetic test kit",
                   menu=[list(p) for p in menu], gamma=gamma), open(os.path.join(pub, "metadata.json"), "w"))
    return pub


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--pairs", default="2,4:3,6:1,5")
    ap.add_argument("--gamma", type=float, default=0.5); ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(); menu = [tuple(int(x) for x in p.split(",")) for p in a.pairs.split(":")]
    print("wrote", build(a.out, menu, a.gamma, a.episodes, a.seed))
