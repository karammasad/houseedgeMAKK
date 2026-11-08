"""Size-state-aware generator (copy of oracle/sim.gen_machine with one change): payout size | win ~ V_T[b_t xor d], b_t = [payout at the size lag >= 2], d = the win-shift direction
(tied, as Thanos learned ~.98). V_T = two oriented size vectors per type estimated from real train histories (V.pkl). Everything else identical to sim.gen_machine."""
import math, pickle, numpy as np
import sys; sys.path.insert(0, "../oracle")
import sim
V = pickle.load(open("V.pkl", "rb")); SIZES = sim.SIZES; SD = sim.SD; cat3 = sim.cat3


def gen_machine(rng, pop, ty, sg, L, size_p, mu_n=None, ls_n=None):
    P = pop[ty]; mu_on, mu_off, s_on, s_off, gam = P["cp"][sg]; mu_n, ls_n = sim.NONE[ty]
    k = rng.choices(range(4), P["pair_p"])[0]; tau = rng.randrange(40, L) if rng.random() < .5 else None; n = L + 1; w = [0] * n; pay = [0] * n
    if k == 3: b = rng.gauss(mu_n, ls_n); r = np.full((2, 3), 1 / (1 + math.exp(-b))); pair = None; d = 0
    else:
        pair = P["pairs"][k]; on = rng.randint(0, 1); d = rng.randint(0, 1); b = [rng.gauss(mu_on, s_on), rng.gauss(mu_off, s_off)]; bs = [b[0] if on == 0 else b[1], b[1] if on == 0 else b[0]]
        r = np.array([[1 / (1 + math.exp(-(bs[s] + gam * SD[d][c]))) for c in range(3)] for s in range(2)])
    for t in range(n):
        if pair is None or t < 7: p = float(r[0][0]) if pair is None else 0.3
        else:
            i, j, kk = pair; g = w[t - i] ^ w[t - j]
            if tau is not None and t >= tau: g = 1 - g
            p = float(r[g][cat3(pay[t - kk])])
        if t < n - 1:
            w[t] = int(rng.random() < p)
            if w[t]:
                st = 0
                if pair is not None and t >= 7: st = int(pay[t - pair[2]] >= 2) ^ d
                pay[t] = rng.choices(SIZES, V[ty][st])[0]
    return dict(type=ty, signal=sg, pair=pair, tau=tau, rates=r, p_true=p, pay=pay[:L], L=L, k=k, d=d)
