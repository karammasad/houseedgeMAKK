"""Synthetic House Edge kits for STRESS-TESTING the round-two pipeline (not for training). Standard library + kit code.

The generator follows the structure reverse-engineered from the main round (adi/FINDINGS, method4 RESEARCH_LOG):
  win at t  : gate bit g = GATE(w[t-i], w[t-j]) XOR polarity; cell 'on' if g = 1; per-machine on/off rates drawn from a
              per-type Beta, shifted on the log-odds scale by signal; the hidden size state also shifts the on/off rate
  size state: payout category (loss / 1x / big) at lag i, split into two groups (partition), random direction
  sizes     : given a win, one of two fixed vectors per type (identical across machines), picked by the size state
  conditions: stable (no flip), drift (each machine flips polarity once, independently), shared_change (all machines of
              the episode flip at the same tick), short_history (40 recent + 24 older ticks observed)
Variants change what the rules say round two may change ("which temporal relationships recur"): see VARIANTS.

  python3 adi/sim/gen.py --variant newlags --out /tmp/sim_newlags [--seed 1]
Writes <out>/public/{train,validation,test}/... and metadata.json in the official format, with validation targets and the
kit's own pooled reference (FrequencyForecaster, strength 20, fitted on this kit's train split).
"""
import argparse, csv, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
TYPES = ("steady", "balanced", "jackpot")
PAYS = (1, 2, 5, 20)
# per type: (on mean, on strength, off mean, off strength)  [main-round learned priors]
RATES = {"steady": (.95, 100, .15, 50), "balanced": (.80, 50, .07, 50), "jackpot": (.60, 25, .02, 100)}
SLOPE = .30
# two size vectors per type [1x, 2x, 5x, 20x]: state 1 (more 1x) / state 0   [learned menu]
SIZES = {"steady": ((.795, .163, .041, .001), (.351, .523, .123, .003)),
         "balanced": ((.499, .342, .136, .023), (.211, .549, .205, .035)),
         "jackpot": ((.378, .226, .298, .098), (.133, .327, .404, .136))}
STATE_SHIFT = {"on": -.8, "off": -.3}   # log-odds shift of the win rate in size state 1 (more 1x -> fewer wins)

BASE = dict(pairs=((1, 3), (2, 5), (4, 7)), gates=("xor",), size_lag="first", partition=((2,), .8, (1,), .1, (0,), .1),
            drift_window=(40, 100), drift_p=1.0, short_flip_p=.3, three_sizes=False, type_mix=(.35, .36, .29))
VARIANTS = {
    "base": {},                                                        # main-like: sanity check of the simulator
    "newlags": dict(pairs=((3, 6), (2, 8), (5, 7))),                   # different recurring lag pairs
    "manylags": dict(pairs=((1, 4), (2, 6), (3, 8), (5, 6), (1, 7), (4, 8))),
    "gates": dict(gates=("xor", "and", "or")),                         # AND / OR gates as well as XOR
    "flips": dict(drift_window=(20, 160), drift_p=.6),                 # reversals earlier/more spread, not every machine
    "sizelag": dict(size_lag="second"),                                # size state at the pair's SECOND lag
    "sizes3": dict(three_sizes=True),                                  # sizes not two-state (one vector per category)
}


def logit(p):
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-x))


def beta(rng, mean, k):
    return min(max(rng.betavariate(mean * k, (1 - mean) * k), 1e-4), 1 - 1e-4)


def cat(v):
    return 0 if v == 0 else (1 if v == 1 else 2)


def gate(form, a, b):
    return a ^ b if form == "xor" else (a & b if form == "and" else a | b)


def machine(rng, cfg, mtype, signal):
    on_m, on_k, off_m, off_k = RATES[mtype]
    on = beta(rng, sig(logit(on_m) + SLOPE * signal), on_k); off = beta(rng, sig(logit(off_m) + SLOPE * signal), off_k)
    i, j = rng.choice(cfg["pairs"]); form = rng.choice(cfg["gates"])
    p = cfg["partition"]; u = rng.random(); grp = p[0] if u < p[1] else (p[2] if u < p[1] + p[3] else p[4])
    k = i if cfg["size_lag"] == "first" else j
    v1, v0 = SIZES[mtype]
    if cfg["three_sizes"]:
        mid = tuple((a + b) / 2 for a, b in zip(v1, v0)); vecs = [v1, mid, v0]; rng.shuffle(vecs)
    else:
        vecs = None
    return dict(on=on, off=off, i=i, j=j, form=form, k=k, grp=grp, dir=rng.random() < .5, v=(v0, v1), vecs=vecs,
                pol=rng.random() < .5, tau=None)


def state_of(m, prev_pay):
    c = cat(prev_pay)
    if m["vecs"] is not None:
        return c
    return int((c in m["grp"]) ^ m["dir"])


def probs(m, w, pay, t):
    """True [p0, p1, p2, p5, p20] at tick t from ticks < t."""
    g = gate(m["form"], w[t - m["i"]], w[t - m["j"]]) ^ m["pol"] ^ (m["tau"] is not None and t >= m["tau"])
    z = state_of(m, pay[t - m["k"]])
    if m["vecs"] is not None:
        sz = m["vecs"][z]; zs = 1 if z == 0 else 0
    else:
        sz = m["v"][z]; zs = z
    base = m["on"] if g else m["off"]
    pw = sig(logit(base) + zs * STATE_SHIFT["on" if g else "off"])
    return [1 - pw] + [pw * x for x in sz]


def draw(rng, p):
    u = rng.random(); acc = 0.0
    for o, x in zip((0, 1, 2, 5, 20), p):
        acc += x
        if u < acc:
            return o
    return 20


def episode(rng, cfg, split, e, cond):
    cut = rng.randint(192, 256); eid = f"sim-{split}-e{e:05d}"
    a_, b_ = cfg["type_mix"][0], cfg["type_mix"][0] + cfg["type_mix"][1]
    types = [TYPES[0 if u < a_ else (1 if u < b_ else 2)] for u in (rng.random() for _ in range(8))]
    lo, hi = cfg["drift_window"]; shared_tau = cut - rng.randint(lo, hi)
    rows = []
    for mi in range(8):
        mtype = types[mi]; signal = rng.choice((-1, 0, 1)); m = machine(rng, cfg, mtype, signal)
        if cond == "drift" and rng.random() < cfg["drift_p"]:
            m["tau"] = cut - rng.randint(lo, hi)
        elif cond == "shared_change":
            m["tau"] = shared_tau
        elif cond == "short_history" and rng.random() < cfg["short_flip_p"]:
            m["tau"] = cut - rng.randint(lo, hi)
        w, pay = [], []
        for t in range(cut + 1):
            if t < 12:
                o = draw(rng, [.5] + [.5 * x for x in m["v"][0]])
            else:
                p = probs(m, w, pay, t)
                if t == cut:
                    target = p
                o = draw(rng, p)
            pay.append(o); w.append(int(o > 0))
        keep = set(range(cut))
        if cond == "short_history":
            keep = set(range(cut - 40, cut)) | set(rng.sample(range(cut - 40), 24))
        mid = f"{eid}-m{mi:03d}"
        rows.append(dict(qid=f"{mid}-next", eid=eid, mid=mid, mtype=mtype, signal=signal, cut=cut, cond=cond,
                         hist=[(t, pay[t]) for t in sorted(keep)], label=pay[cut], target=target))
    return rows


def write_split(pub, split, rows):
    d = os.path.join(pub, split); os.makedirs(d, exist_ok=True)
    def w(name, header, data):
        with open(os.path.join(d, name), "w", newline="") as f:
            c = csv.writer(f); c.writerow(header); c.writerows(data)
    w("queries.csv", ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"],
      [(r["qid"], r["eid"], r["mid"], r["mtype"], r["signal"], r["cut"]) for r in rows])
    w("history.csv", ["episode_id", "machine_id", "tick", "payout"],
      [(r["eid"], r["mid"], t, v) for r in rows for t, v in r["hist"]])
    w("sample_submission.csv", ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], [(r["qid"],) + (.2,) * 5 for r in rows])
    if split != "test":
        w("labels.csv", ["query_id", "outcome"], [(r["qid"], r["label"]) for r in rows])
        w("conditions.csv", ["query_id", "condition"], [(r["qid"], r["cond"]) for r in rows])
    if split == "validation":
        w("targets.csv", ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"],
          [(r["qid"], r["eid"], *[repr(x) for x in r["target"]], r["label"]) for r in rows])


def reference(pub):
    sys.path.insert(0, os.path.join(ROOT, "kit"))
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import load_inputs, read_labels
    q, h = load_inputs(os.path.join(pub, "train")); lab = read_labels(os.path.join(pub, "train", "labels.csv"), q)
    m = FrequencyForecaster(method="pooled", prior_strength=20.0).fit(q, h, lab)
    vq, vh = load_inputs(os.path.join(pub, "validation")); pred = m.predict(vq, vh)
    with open(os.path.join(pub, "validation", "reference.csv"), "w", newline="") as f:
        c = csv.writer(f); c.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for k in sorted(pred):
            c.writerow([k] + [format(x, ".17g") for x in pred[k]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="base", choices=sorted(VARIANTS))
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); cfg = dict(BASE, **VARIANTS[a.variant]); pub = os.path.join(a.out, "public")
    rng = random.Random(f"{a.variant}|{a.seed}")
    conds = ("stable", "short_history", "drift", "shared_change")
    for split, n in (("train", 24), ("validation", 24), ("test", 48)):
        rows = []; e = 0
        for c in conds:
            for _ in range(n):
                rows += episode(rng, cfg, split, e, c); e += 1
        write_split(pub, split, rows)
    reference(pub)
    meta = json.load(open(os.path.join(ROOT, "data", "main", "metadata.json")))   # official layout (checker needs it)
    meta.update({"round": "adaptation", "season_id": __import__("hashlib").sha256(f"sim-{a.variant}-{a.seed}".encode()).hexdigest()[:24],
                 "sim_variant": a.variant, "max_points": 40,
                 "note": "SYNTHETIC stress-test kit (adi/sim/gen.py), not organizer data"})
    with open(os.path.join(pub, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", pub)


if __name__ == "__main__":
    main()
