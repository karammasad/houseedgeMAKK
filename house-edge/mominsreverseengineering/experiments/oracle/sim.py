"""Training-only simulator with KNOWN latent truth, built from the fitted shared-base population (experiments/gm_sb.pkl, train-only fit).
Truth for a machine: type, signal, pair (i,j,k) in the three fitted families or 'none', which XOR side is the 'on' side, base logits
~ logit-normal(mu(signal), sigma) per side, size-state direction d, gamma, optional polarity flip at tau (prob .5, uniform in [40, L)), payout sizes
iid from the type's empirical distribution of wins. Rates of a machine: r[side][cat] = sigmoid(b_side + gamma * SD[d][cat]) (the xsg cells).
Histories: long = all L ticks; short = last 40 ticks + 24 random older ticks (the real short layout). Target = tick L (truth probability known).
Misspecification is deliberately ABSENT: results measure inference loss inside the model family, not model error.
"""
import csv, os, pickle, random, math, zlib
import numpy as np

SD = np.array([[-1., -1., 1.], [1., 1., -1.]]); SIZES = (1, 2, 5, 20); TYPES = ("steady", "balanced", "jackpot")
_gm = pickle.load(open(os.path.join(os.path.dirname(__file__), "..", "gm_sb.pkl"), "rb"))


def population():
    pop = {}
    mass = {}
    for st, p in zip(_gm.ST, _gm.sprior):
        f, l = st
        if len(l) >= 2 and f in ("xor", "xs", "xsg"):
            mass[tuple(l[:2])] = mass.get(tuple(l[:2]), 0) + p
    top = sorted(mass, key=lambda x: -mass[x])[:3]
    klag = {(1, 3): 1, (2, 5): 2, (4, 7): 4}
    pairs = [(a, b, klag.get((a, b), a)) for a, b in top]
    w = np.array([mass[(a, b)] for a, b, _ in pairs]); w = w / w.sum() * 0.95
    for t in TYPES:
        cp = {sg: [float(x) for x in _gm.cellpriors[(t, sg)][5:10]] for sg in (-1, 0, 1)}
        pop[t] = dict(cp=cp, pairs=pairs, pair_p=list(w) + [0.05])
    return pop


def cat3(v): return 0 if v == 0 else (1 if v == 1 else 2)


NONE = {t: (float(v['mu_n']), float(v['ls_n'])) for t, v in _gm.shared_fit['per_type'].items()}   # REAL fitted no-gate rate distribution per type (extreme rates)


def gen_machine(rng, pop, ty, sg, L, size_p):
    mu_n, ls_n = NONE[ty]
    P = pop[ty]; mu_on, mu_off, s_on, s_off, gam = P["cp"][sg]
    k = rng.choices(range(4), P["pair_p"])[0]
    tau = rng.randrange(40, L) if rng.random() < .5 else None
    n = L + 1; w = [0] * n; pay = [0] * n
    if k == 3:                                                             # no gate: one base rate
        b = rng.gauss(mu_n, ls_n); r = np.full((2, 3), 1 / (1 + math.exp(-b))); pair = None; on = d = 0
    else:
        pair = P["pairs"][k]; on = rng.randint(0, 1); d = rng.randint(0, 1)
        b = [rng.gauss(mu_on, s_on), rng.gauss(mu_off, s_off)]; bs = [b[0] if on == 0 else b[1], b[1] if on == 0 else b[0]]
        r = np.array([[1 / (1 + math.exp(-(bs[s] + gam * SD[d][c]))) for c in range(3)] for s in range(2)])
    for t in range(n):
        if pair is None or t < 7:
            p = float(r[0][0]) if pair is None else 0.3; ptrue = p
        else:
            i, j, kk = pair; g = w[t - i] ^ w[t - j]
            if tau is not None and t >= tau: g = 1 - g
            p = float(r[g][cat3(pay[t - kk])]); ptrue = p
        if t < n - 1:
            w[t] = int(rng.random() < p)
            pay[t] = rng.choices(SIZES, size_p[ty])[0] if w[t] else 0
    return dict(type=ty, signal=sg, pair=pair, tau=tau, rates=r, p_true=ptrue, pay=pay[:L], L=L, k=k)


def size_dist():
    from common.data import public_dir, load_split
    q, seq = load_split(public_dir(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "main")), "train")
    c = {t: np.ones(4) for t in TYPES}
    for r in q:
        for v in seq[r["machine_id"]]:
            if v > 0: c[r["machine_type"]][SIZES.index(v)] += 1
    return {t: list(x / x.sum()) for t, x in c.items()}


def make_world(seed, n_train=768, n_test=900, size_p=None):
    rng = random.Random(seed); pop = population(); size_p = size_p or size_dist()
    def batch(n, tag):
        out = []
        for i in range(n):
            ty = TYPES[i % 3]; sg = rng.choice((-1, 0, 1)); L = rng.randint(192, 256); m = gen_machine(rng, pop, ty, sg, L, size_p)
            short = (i % 4 == 3)
            keep = None
            if short:
                keep = set(range(L - 40, L)) | set(rng.sample(range(0, L - 40), 24))
            m["hist"] = [v if keep is None or t in keep else -1 for t, v in enumerate(m["pay"])]
            m["short"] = short; m["id"] = f"{tag}{seed}_{i}"; m["ep"] = f"{tag}{seed}_e{i // 8}"; out.append(m)
        return out
    return dict(train=batch(n_train, "tr"), test=batch(n_test, "te"), pop=pop, size_p=size_p)


def write_pub(machines, pub):
    os.makedirs(os.path.join(pub, "train"), exist_ok=True)
    with open(os.path.join(pub, "train", "queries.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"])
        for m in machines: w.writerow([m["id"], m["ep"], m["id"], m["type"], m["signal"], m["L"]])
    with open(os.path.join(pub, "train", "history.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["machine_id", "tick", "payout"])
        for m in machines:
            for t, v in enumerate(m["hist"]):
                if v >= 0: w.writerow([m["id"], t, v])
