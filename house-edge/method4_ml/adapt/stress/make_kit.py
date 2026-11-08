"""Synthetic stress kits (Person D): a generic slot simulator with KNOWN true probabilities and switchable mechanisms, written to a
kit-shaped public/ folder so the full pipeline (fit, predict, common.evaluate.points) runs on it unchanged. Used to check that the
model adapts to round-two-like changes (it is never fit on main-round truth, so it cannot overfit the main round).

  python -m method4_ml.adapt.stress.make_kit --kind cross --out /tmp/stress_cross

Kinds
  main    own-history XOR pairs / singles (lags <= 8), machine-specific polarity, 3 types, short logs = last 40 + 24 older
  cross   half the machines are driven by a NEIGHBOUR's win at a lag (cross-machine rule), a 4th machine type ('wild')
  triple  a third of the machines use an XOR triple of own lags; short logs = last 30 + every 4th older tick (new shape)
  sizes   win gate = own payout category (big win) at a lag, i.e. a size-driven rule; signal takes 5 values (-2..2)
Every kind: stable / short_history / drift (one machine flips polarity 40-100 ticks before the cutoff) / shared_change (all machines
flip together); sizes given a win switch between 2 vectors per type by the payout category at a per-machine lag.
"""
import argparse, csv, json, math, os, random

PAY = (0, 1, 2, 5, 20)
CONDS = ("stable", "short_history", "drift", "shared_change")


def sizes_for(types, rng):
    out = {}
    for t in types:
        a = [rng.uniform(.4, .7), rng.uniform(.15, .3), rng.uniform(.05, .15), rng.uniform(.01, .05)]
        b = [rng.uniform(.15, .35), rng.uniform(.3, .5), rng.uniform(.1, .25), rng.uniform(.02, .08)]
        out[t] = ([x / sum(a) for x in a], [x / sum(b) for x in b])
    return out


def machine_rule(kind, m, M, rng):
    """Returns a rule: list of (machine_index, lag, what) bits XORed; what = 'win' or 'big'."""
    own = lambda k: [(m, l, "win") for l in sorted(rng.sample(range(1, 9), k))]
    if kind == "cross" and rng.random() < .5:
        j = rng.choice([x for x in range(M) if x != m]); return [(j, rng.randint(1, 8), "win")]
    if kind == "triple" and rng.random() < 1 / 3:
        return own(3)
    if kind == "sizes" and rng.random() < .5:
        return [(m, rng.randint(1, 8), "big")]
    return own(2 if rng.random() < .7 else 1)


def episode(kind, cond, eid, cut, types, menus, rng, M=8):
    tys = [rng.choice(types) for _ in range(M)]
    sig = [rng.choice((-2, -1, 0, 1, 2) if kind == "sizes" else (-1, 0, 1)) for _ in range(M)]
    rules = [machine_rule(kind, m, M, rng) for m in range(M)]
    base_on = {t: .55 + .1 * i for i, t in enumerate(types)}
    on = [min(.95, max(.2, rng.gauss(base_on[tys[m]] + .05 * sig[m], .1))) for m in range(M)]
    off = [min(.4, max(.02, rng.gauss(.12, .06))) for _ in range(M)]
    pol = [rng.random() < .5 for _ in range(M)]                    # machines of the same type can respond oppositely
    slag = [rng.randint(1, 8) for _ in range(M)]
    tau = [None] * M
    if cond == "drift":
        k = rng.randrange(M); tau[k] = cut - rng.randint(40, 100)
    elif cond == "shared_change":
        t0 = cut - rng.randint(40, 100); tau = [t0] * M
    P = [[0] * (cut + 1) for _ in range(M)]; truth = None
    for t in range(cut + 1):
        probs = []
        for m in range(M):
            b = 0
            for (j, l, what) in rules[m]:
                v = P[j][t - l] if t - l >= 0 else 0
                b ^= int(v > 0) if what == "win" else int(v >= 2)
            flip = pol[m] ^ (tau[m] is not None and t >= tau[m])
            pw = on[m] if (b ^ flip) else off[m]
            V1, V2 = menus[tys[m]]; prev = P[m][t - slag[m]] if t - slag[m] >= 0 else 0
            sz = V1 if prev == 1 else V2
            probs.append([1 - pw] + [pw * x for x in sz])
        if t == cut:
            truth = probs; break
        for m in range(M):
            P[m][t] = rng.choices(PAY, probs[m])[0]
    return tys, sig, [p[:cut] for p in P], truth


def mask(cut, shape, rng):
    if shape == "classic":
        keep = set(range(cut - 40, cut)) | set(rng.sample(range(cut - 40), 24))
    else:   # last 30 + every 4th older tick (a different short-log shape)
        keep = set(range(cut - 30, cut)) | set(range(rng.randrange(4), cut - 30, 4))
    return keep


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def make(kind, out, n_train=96, n_val=96, n_test=48, seed=0):
    rng = random.Random(f"{kind}|{seed}")
    types = ["steady", "balanced", "jackpot"] + (["wild"] if kind == "cross" else [])
    menus = sizes_for(types, rng); shape = "stride" if kind == "triple" else "classic"
    data = {}
    for split, n in (("train", n_train), ("validation", n_val), ("test", n_test)):
        Q, H, LBL, CND, TGT = [], [], [], [], []
        for e in range(n):
            cond = CONDS[e % 4]; cut = rng.randint(192, 256); eid = f"{kind}-{split}-e{e:05d}"
            tys, sig, P, truth = episode(kind, cond, eid, cut, types, menus, rng)
            for m in range(8):
                mid = f"{eid}-m{m:03d}"; qid = mid + "-next"
                keep = mask(cut, shape, rng) if cond == "short_history" else set(range(cut))
                Q.append([qid, eid, mid, tys[m], sig[m], cut])
                H += [[eid, mid, t, P[m][t]] for t in range(cut) if t in keep]
                y = rng.choices(PAY, truth[m])[0]; LBL.append([qid, y]); CND.append([qid, cond])
                TGT.append([qid, eid] + truth[m] + [y])
        d = os.path.join(out, split)
        write(os.path.join(d, "queries.csv"), ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"], Q)
        write(os.path.join(d, "history.csv"), ["episode_id", "machine_id", "tick", "payout"], H)
        if split != "test":
            write(os.path.join(d, "labels.csv"), ["query_id", "outcome"], LBL)
            write(os.path.join(d, "conditions.csv"), ["query_id", "condition"], CND)
        if split == "validation":
            write(os.path.join(d, "targets.csv"), ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"], TGT)
        data[split] = (Q, H, LBL)
    # frozen reference = pooled starter: machine counts shrunk to the train type frequencies with strength 20
    Q, H, L = data["train"]; tq = {q[2]: q[3] for q in Q}; tf = {t: [1.0] * 5 for t in types}
    for _, mid, _, p in H:
        tf[tq[mid]][PAY.index(p)] += 1
    tf = {t: [x / sum(v) for x in v] for t, v in tf.items()}
    Q, H, _ = data["validation"]; cnt = {q[2]: [0] * 5 for q in Q}
    for _, mid, _, p in H:
        cnt[mid][PAY.index(p)] += 1
    ref = []
    for q in Q:
        c = cnt[q[2]]; n = sum(c); ref.append([q[0]] + [(c[i] + 20 * tf[q[3]][i]) / (n + 20) for i in range(5)])
    write(os.path.join(out, "validation", "reference.csv"), ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], ref)
    json.dump({"round": "stress-" + kind, "max_points": 40}, open(os.path.join(out, "metadata.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args(); make(a.kind, a.out, seed=a.seed); print("wrote", a.out)
