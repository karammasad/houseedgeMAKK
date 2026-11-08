"""Known-parameter comparison on SYNTHETIC episodes from the fixed chain (ground truth available): exact forward-backward decoder vs the
production-style local approximation, both given the TRUE machine parameters, at query-like cutoffs (45-58% of a 220-tick log) under the
real mask layouts. Parameters are drawn from the fitted shared-base population (gm_sb cellpriors, per type). Metric: squared distance of the
3-category predictive distribution from the TRUE one (no outcome noise). Episode-level bootstrap = replicate-level bootstrap here."""
import pickle, random, sys, time, zlib
import numpy as np
from hmm12 import Chain, exact_predict, local_predict, hazard

gm = pickle.load(open("../gm_sb.pkl", "rb")); SD = np.array([[-1., -1., 1.], [1., 1., -1.]])
L = 220; PC = 0.5; MASKS = ("n40", "short64", "n80", "full")


def draw(rng, ty):
    mu_on, mu_off, s_on, s_off, gam = [float(x) for x in gm.cellpriors[(ty, 0)][5:10]]
    b = [rng.gauss(mu_on, s_on), rng.gauss(mu_off, s_off)]
    if rng.random() < .5: b = b[::-1]
    d = rng.randint(0, 1); beta = rng.uniform(.3, .6); wbar = .3
    r = np.array([[1 / (1 + np.exp(-(b[g] + gam * SD[d][c]))) for c in range(3)] for g in range(2)])
    return Chain(r, beta, [1 - wbar, wbar * (1 - beta), wbar * beta])


def simulate(rng, ch, u):
    tau = rng.randrange(40, L) if rng.random() < .5 else None
    c = [rng.choices(range(3), ch.m)[0] for _ in range(3)]
    for t in range(3, u + 1):
        pol = 0 if tau is None or t < tau else 1
        g = int(c[t - 1] > 0) ^ int(c[t - 3] > 0); p = ch.p_cat(pol, c[t - 1], int(c[t - 2] > 0), int(c[t - 3] > 0))
        if t == u:
            return c, p, tau
        c.append(rng.choices(range(3), p)[0])


def keep(mask, t, rng):
    if mask == "full": return None
    if mask.startswith("n"): return set(range(max(0, t - int(mask[1:])), t))
    return set(range(t - 40, t)) | set(rng.sample(range(0, t - 40), 24))


def job(args):
    ty, seed = args; rng = random.Random(seed); ch = draw(rng, ty); u = rng.randrange(int(.45 * L), int(.58 * L))
    c, p_true, tau = simulate(rng, ch, u); out = {}
    for mk in MASKS:
        k = keep(mk, u, random.Random(seed * 7 + len(mk))); obs = [v if k is None or i in k else -1 for i, v in enumerate(c)]
        t0 = time.perf_counter(); pl = local_predict(ch, obs, u, PC); t1 = time.perf_counter(); pe, _ = exact_predict(ch, obs, u, PC); t2 = time.perf_counter()
        out[mk] = (float(((pl - p_true) ** 2).sum()), float(((pe - p_true) ** 2).sum()), t1 - t0, t2 - t1, float((pl[0] - pe[0])))
    return ty, tau is not None and tau <= u, out


if __name__ == "__main__":
    from multiprocessing import Pool
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    jobs = [(ty, 1000 * i + zlib.crc32(ty.encode()) % 997) for ty in ("steady", "balanced", "jackpot") for i in range(N)]
    with Pool(8) as p: res = p.map(job, jobs, chunksize=50)
    rng = random.Random(1)
    def boot(d):
        d = np.array(d); ms = sorted(d[rng.choices(range(len(d)), k=len(d))].mean() for _ in range(1000)); return d.mean(), ms[25], ms[974]
    print(f"{len(res)} synthetic machines (N={N}/type); MSE vs TRUE 3-cat distribution, x1e-4; paired difference exact - local with 95% bootstrap CI")
    print(f"{'subset':26s}{'mask':9s}{'local':>9s}{'exact':>9s}{'diff (exact-local)':>30s}{'ms/pred local':>15s}{'ms/pred exact':>15s}")
    for name, sel in (("ALL", lambda r: True), ("steady", lambda r: r[0] == "steady"), ("balanced", lambda r: r[0] == "balanced"), ("jackpot", lambda r: r[0] == "jackpot"),
                      ("flipped before target", lambda r: r[1]), ("no flip before target", lambda r: not r[1])):
        rs = [r for r in res if sel(r)]
        for mk in MASKS:
            l = [r[2][mk][0] * 1e4 for r in rs]; e = [r[2][mk][1] * 1e4 for r in rs]; df = [b - a for a, b in zip(l, e)]; m, lo, hi = boot(df)
            print(f"{name:26s}{mk:9s}{np.mean(l):9.2f}{np.mean(e):9.2f}{m:+13.3f} [{lo:+.3f},{hi:+.3f}]{np.mean([r[2][mk][2] for r in rs])*1e3:15.2f}{np.mean([r[2][mk][3] for r in rs])*1e3:15.2f}")
    pickle.dump(res, open("synth_res.pkl", "wb"))
