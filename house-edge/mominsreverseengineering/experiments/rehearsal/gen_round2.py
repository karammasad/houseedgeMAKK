"""FRESH synthetic Round 2 kit (one PREDEFINED configuration, written before any model saw it; seed 20261108). Not previously used or inspected.
Kit rules honoured: lags <= 8, changed recurring lag structures, win chances and sizes shifted, four conditions, 40+24 short layout, 8 machines/episode,
episode counts per condition train 24 / validation 24 / test 48. Hidden truth (true probability vectors, realized outcomes) is written OUTSIDE the public kit.
Config (frozen):
  lag structures (per machine, mixture): pair (1,5) size lag 5 [.25] | pair (2,3) size lag 2 [.25] | pair (3,8) size lag 8 [.20] | pair (4,6) size lag 4 [.20] | single lag (3,) size lag 3 [.05] | no gate [.05]
  (none of Round 1's (1,3), (2,5), (4,7); size lag = smaller lag for two pairs, the LARGER lag for two, the gate lag for the single)
  rates: per-type logit means shifted vs Round 1: on +0.25, off -0.30; sigma_on .34, sigma_off .50; signal slope .40; size-state shift gamma = 1.2 x main (.56 / .64 / .36)
  sizes: two vectors per type = 0.8 x main vector + 0.2 x a shifted alternative (heavier 2x/5x mass)
  flips: stable none; short_history none; drift: per-machine tau/cutoff = episode mean U[.50,.78] + N(0,.05); shared_change: one tau/cutoff U[.52,.74] per episode
  no-gate machines: single rate from the Round 1 per-type no-gate distribution."""
import csv, json, math, os, random, sys
import numpy as np

SD = {0: (-1, -1, 1), 1: (1, 1, -1)}; PAY = (1, 2, 5, 20); COND = ("stable", "short_history", "drift", "shared_change"); TYPES = ("steady", "balanced", "jackpot")
MAIN = dict(steady=dict(on=2.9, off=-1.8, gam=.30, V=([.743, .202, .054, .001], [.456, .429, .113, .002]), none=(5.0, .3)),
            balanced=dict(on=1.75, off=-3.0, gam=.46, V=([.49, .354, .135, .021], [.217, .543, .204, .036]), none=(-.1, 1.3)),
            jackpot=dict(on=.38, off=-4.4, gam=.53, V=([.312, .273, .322, .094], [.134, .326, .39, .15]), none=(-5.9, .35)))
ALT = ([.30, .38, .27, .05], [.15, .40, .33, .12])
STRUCT = [((1, 5), 5, .25), ((2, 3), 2, .25), ((3, 8), 8, .20), ((4, 6), 4, .20), ((3,), 3, .05), (None, 1, .05)]
sig = lambda x: 1 / (1 + math.exp(-x)); cat3 = lambda v: 0 if v == 0 else (1 if v == 1 else 2)


def vecs(ty):
    a = MAIN[ty]["V"]; return [list(np.array(a[s]) * .8 + np.array(ALT[s]) * .2) for s in (0, 1)]


def machine(rng, ty, sg, cutoff, tau, struct):
    P = MAIN[ty]; gates, kk, _ = struct; V = vecs(ty); gam = P["gam"] * 1.2
    mu_on = P["on"] + .25 + .40 * sg; mu_off = P["off"] - .30 + .40 * sg
    if gates is None: r_none = sig(rng.gauss(*P["none"]))
    else:
        b = [rng.gauss(mu_on, .34), rng.gauss(mu_off, .50)]; on = rng.randint(0, 1); bs = (b[0], b[1]) if on == 0 else (b[1], b[0]); d = rng.randint(0, 1)
    w = []; pay = []
    def prob(t):
        if gates is None: return r_none, V[0]
        if t < 8: return .3, V[0]
        g = 0
        for l in gates: g ^= (pay[t - l] > 0)
        if tau is not None and t >= tau: g ^= 1
        c = cat3(pay[t - kk]); s = SD[d][c]; st = int(pay[t - kk] >= 2) ^ d
        return sig(bs[g] + gam * s), V[st]
    for t in range(cutoff):
        p, v = prob(t); pay.append(0 if rng.random() > p else rng.choices(PAY, v)[0])
    p, v = prob(cutoff); truth = [1 - p] + [p * x for x in v]; out = 0 if rng.random() > p else rng.choices(PAY, v)[0]
    return pay, truth, out


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f: w = csv.writer(f); w.writerow(header); w.writerows(rows)


def build(out, hidden, seed=20261108):
    rng = random.Random(seed); pub = os.path.join(out, "public"); counts = {"train": 24, "validation": 24, "test": 48}; pooled = {}; probs = [s[2] for s in STRUCT]
    for split, neps in counts.items():
        Q, H, Lb, Cn, Tg = [], [], [], [], []; e = 0
        for cond in COND:
            for _ in range(neps):
                eid = f"r2syn-{split}-e{e:05d}"; e += 1; cutoff = rng.randint(192, 256)
                ep_tau = rng.uniform(.52, .74) if cond == "shared_change" else (rng.uniform(.50, .78) if cond == "drift" else None)
                for m in range(8):
                    mid = f"{eid}-m{m:03d}"; ty = TYPES[(m + e) % 3]; sg = rng.choice((-1, 0, 1)); st = rng.choices(STRUCT, probs)[0]
                    tau = None
                    if cond == "shared_change": tau = int(cutoff * ep_tau)
                    elif cond == "drift": tau = int(cutoff * min(.85, max(.4, ep_tau + rng.gauss(0, .05))))
                    pay, truth, outc = machine(rng, ty, sg, cutoff, tau, st); qid = f"{mid}-next"; Q.append([qid, eid, mid, ty, sg, cutoff])
                    keep = range(cutoff)
                    if cond == "short_history": keep = sorted(set(range(cutoff - 40, cutoff)) | set(rng.sample(range(0, cutoff - 40), 24)))
                    H += [[eid, mid, t, pay[t]] for t in keep]; Lb.append([qid, outc]); Cn.append([qid, cond]); Tg.append([qid, eid, cond, ty] + truth + [outc])
                    if split == "train":
                        for v in pay: pooled.setdefault(ty, [0] * 5)[(0, 1, 2, 5, 20).index(v)] += 1
        d = os.path.join(pub, split)
        write(os.path.join(d, "queries.csv"), ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"], Q)
        write(os.path.join(d, "history.csv"), ["episode_id", "machine_id", "tick", "payout"], H)
        if split != "test": write(os.path.join(d, "labels.csv"), ["query_id", "outcome"], Lb); write(os.path.join(d, "conditions.csv"), ["query_id", "condition"], Cn)
        if split == "validation":
            write(os.path.join(d, "targets.csv"), ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"], [[t[0], t[1]] + t[4:] for t in Tg])
            write(os.path.join(d, "reference.csv"), ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], [[r[0]] + [x / sum(pooled[r[3]]) for x in pooled[r[3]]] for r in Q])
        write(os.path.join(hidden, f"truth_{split}.csv"), ["query_id", "episode_id", "condition", "type", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"], Tg)
    json.dump(dict(round="adaptation", max_points=40, synthetic=True, season_id="0" * 24, suite_version="experimentation-suite-v3", counts_per_condition=counts,
                   machines_per_episode=8, note="fresh synthetic Round 2 rehearsal"), open(os.path.join(pub, "metadata.json"), "w"))
    return pub

if __name__ == "__main__":
    print("wrote", build(sys.argv[1], sys.argv[2]))
