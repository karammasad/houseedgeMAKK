"""Mock Round-2 (adaptation) kits in the official file layout, with KNOWN true probabilities, to test how thanos adapts.

NOT a claim about the real round 2. The base generator is the main-round one as measured by the team (per-type on/off win logits,
signal slope .34, per-machine rate noise, a binary size state that shifts the win logit by +-gamma and picks one of two size vectors
per type, one polarity flip in drift / shared_change, short logs = 40 recent + 24 older ticks; constants from
mominsreverseengineering/experiments/synth_kit.py). Each scenario changes only what the kit says round 2 changes: "which temporal
relationships recur", plus the things the rules say may matter ("neighbouring machines can help", "win chances and amounts can change").

  python3 -m thanos.mock_round2 --scenario all --out build/mock_r2
Scenarios
  control    main-round relationships: XOR pairs (1,3) (2,5) (4,7) + 5% no gate      (does the new code leave main-like data alone?)
  shifted    new own lag sets: pairs (2,6) (3,8), single (5), triple (1,4,7)          (the most literal reading of "changed patterns")
  neighbour  half the machines: own lag 1 XOR the NEXT machine's win at lag 2 / alone at lag 3; rest own pairs (2,6) (3,8)
  hard       everything at once: triples, neighbour (random partner), a size-driven gate (own payout >= 2 at lag 3), size state reads
             "any win" instead of "payout >= 2", and flips earlier (tau / cutoff .35-.55 instead of .62-.76)
Every machine is simulated synchronously, so cross-machine inputs are real.
"""
import argparse, csv, json, math, os, random

TYPES = {"steady": dict(on=2.9, off=-1.8, V=([.455, .436, .107, .002], [.754, .197, .048, .001])),
         "balanced": dict(on=1.7, off=-3.0, V=([.5, .348, .13, .022], [.214, .547, .205, .034])),
         "jackpot": dict(on=0.4, off=-4.4, V=([.335, .245, .315, .105], [.121, .324, .416, .139]))}
PAY = (0, 1, 2, 5, 20); COND = ("stable", "short_history", "drift", "shared_change")
GAMMA, SLOPE, SIGMA = .45, .34, .3
sig = lambda x: 1 / (1 + math.exp(-x))
SIZEDEF = {"big": lambda v: v >= 2, "win": lambda v: v >= 1}

# A rule = (inputs, size_lag); input = ("w", lag) own win | ("b", lag) own payout >= 2 | ("n", r, lag) win of machine (m + r) % 8
SCEN = {
    "control": dict(menu=[(.32, [("w", 1), ("w", 3)], 1), (.32, [("w", 2), ("w", 5)], 2), (.31, [("w", 4), ("w", 7)], 4), (.05, [], 3)],
                    sizedef="big", flip=(.62, .76)),
    "shifted": dict(menu=[(.3, [("w", 2), ("w", 6)], 2), (.3, [("w", 3), ("w", 8)], 3), (.2, [("w", 5)], 5),
                          (.2, [("w", 1), ("w", 4), ("w", 7)], 1)], sizedef="big", flip=(.62, .76)),
    "neighbour": dict(menu=[(.3, [("w", 1), ("n", 1, 2)], 1), (.2, [("n", 1, 3)], 2), (.25, [("w", 2), ("w", 6)], 2),
                            (.25, [("w", 3), ("w", 8)], 3)], sizedef="big", flip=(.62, .76)),
    "hard": dict(menu=[(.25, [("w", 1), ("w", 4), ("w", 7)], 1), (.25, [("w", 2), ("w", 7)], 2), (.25, [("n", None, 2)], 3),
                       (.25, [("b", 3)], 3)], sizedef="win", flip=(.35, .55)),
}


def episode(sc, cond, cut, rng):
    S = SCEN[sc]; M = 8; ms = []
    tau_sh = int(cut * rng.uniform(*S["flip"])) if cond == "shared_change" else None
    for m in range(M):
        ty = rng.choice(list(TYPES)); s = rng.choice((-1, 0, 1)); p = TYPES[ty]
        u = rng.random(); acc = 0
        for w_, ins, sl in S["menu"]:
            acc += w_
            if u < acc:
                break
        ins = [("n", rng.randint(1, M - 1), x[2]) if x[0] == "n" and x[1] is None else x for x in ins]
        flip = tau_sh if cond == "shared_change" else (int(cut * rng.uniform(*S["flip"])) if cond == "drift" else None)
        ms.append(dict(ty=ty, s=s, ins=ins, sl=sl, pol=rng.randrange(2), dir=rng.randrange(2), flip=flip,
                       on=p["on"] + SLOPE * s + rng.gauss(0, SIGMA), off=p["off"] + SLOPE * s + rng.gauss(0, SIGMA)))
    pay = [[] for _ in range(M)]; sd = SIZEDEF[S["sizedef"]]
    get = lambda j, t: pay[j][t] if t >= 0 else 0
    for t in range(cut + 1):
        probs = []
        for m, d in enumerate(ms):
            g = 0
            for x in d["ins"]:
                if x[0] == "w":
                    g ^= int(get(m, t - x[1]) > 0)
                elif x[0] == "b":
                    g ^= int(get(m, t - x[1]) >= 2)
                else:
                    g ^= int(get((m + x[1]) % M, t - x[2]) > 0)
            on = (g ^ d["pol"] ^ int(d["flip"] is not None and t >= d["flip"])) if d["ins"] else 1
            st = int(sd(get(m, t - d["sl"]))) ^ d["dir"]
            base = d["on"] if on else d["off"]
            if not d["ins"]:
                base = .5 * (d["on"] + d["off"])
            pw = sig(base + GAMMA * (2 * st - 1)); V = TYPES[d["ty"]]["V"][st]
            probs.append([1 - pw] + [pw * v for v in V])
        if t == cut:
            return ms, pay, probs
        for m in range(M):
            pay[m].append(rng.choices(PAY, probs[m])[0])


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def build(sc, out, n=(96, 96, 16), seed=0):
    rng = random.Random(f"{sc}|{seed}"); pub = os.path.join(out, sc, "public"); pooled = {t: [1.0] * 5 for t in TYPES}; data = {}
    for split, ne in zip(("train", "validation", "test"), n):
        Q, Hh, L, C, T = [], [], [], [], []
        for e in range(ne):
            cond = COND[e % 4]; cut = rng.randint(192, 256); eid = f"r2{sc[:2]}-{split[:2]}-e{e:04d}"
            ms, pay, probs = episode(sc, cond, cut, rng)
            for m, d in enumerate(ms):
                mid = f"{eid}-m{m:02d}"; qid = mid + "-q"; y = rng.choices(PAY, probs[m])[0]
                keep = sorted(set(range(cut - 40, cut)) | set(rng.sample(range(cut - 40), 24))) if cond == "short_history" else range(cut)
                Q.append([qid, eid, mid, d["ty"], d["s"], cut]); Hh += [[eid, mid, t, pay[m][t]] for t in keep]
                L.append([qid, y]); C.append([qid, cond]); T.append([qid, eid] + probs[m] + [y])
                if split == "train":
                    for t in keep:
                        pooled[d["ty"]][PAY.index(pay[m][t])] += 1
        data[split] = (Q, Hh)
        dd = os.path.join(pub, split)
        write(os.path.join(dd, "queries.csv"), ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"], Q)
        write(os.path.join(dd, "history.csv"), ["episode_id", "machine_id", "tick", "payout"], Hh)
        if split != "test":
            write(os.path.join(dd, "labels.csv"), ["query_id", "outcome"], L); write(os.path.join(dd, "conditions.csv"), ["query_id", "condition"], C)
        if split == "validation":
            write(os.path.join(dd, "targets.csv"), ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"], T)
    # frozen reference stand-in = pooled starter: the machine's own payout counts shrunk to its type's train frequencies (strength 20)
    tf = {t: [x / sum(v) for x in v] for t, v in pooled.items()}; Q, Hh = data["validation"]; cnt = {q[2]: [0] * 5 for q in Q}
    for _, mid, _, p in Hh:
        cnt[mid][PAY.index(p)] += 1
    ref = [[q[0]] + [(cnt[q[2]][i] + 20 * tf[q[3]][i]) / (sum(cnt[q[2]]) + 20) for i in range(5)] for q in Q]
    write(os.path.join(pub, "validation", "reference.csv"), ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], ref)
    json.dump(dict(round="adaptation", max_points=40, synthetic=True, scenario=sc, seed=seed), open(os.path.join(pub, "metadata.json"), "w"))
    return pub


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", default="all"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    for sc in (SCEN if a.scenario == "all" else a.scenario.split(",")):
        print("wrote", build(sc, a.out, seed=a.seed))
