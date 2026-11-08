"""Brute-force validation of hmm12.exact_predict on tiny synthetic sequences (random parameters, masks, pc)."""
import random, numpy as np
from hmm12 import Chain, exact_predict, local_predict, brute_predict
rng = random.Random(7); worst_p = worst_j = 0; n = 0; lw = 0
for trial in range(300):
    r = np.array([[rng.uniform(.02, .95) for _ in range(3)] for _ in range(2)]); beta = rng.uniform(.1, .7)
    wbar = rng.uniform(.1, .6); m = np.array([1 - wbar, wbar * (1 - beta), wbar * beta]); ch = Chain(r, beta, m)
    u = rng.randint(6, 9); obs = [rng.choice(range(3)) for _ in range(u)]
    pc = rng.choice([.05, .5, .9])
    for t in rng.sample(range(u), rng.randint(0, min(5, u))):          # mask some ticks (target inputs u-1, u-3 stay observed for the local check)
        if t not in (u - 1, u - 3):
            obs[t] = -1
    p_e, j_e = exact_predict(ch, obs, u, pc); p_b, j_b = brute_predict(ch, obs, u, pc)
    worst_p = max(worst_p, float(np.abs(p_e - p_b).max())); worst_j = max(worst_j, abs(j_e - j_b) / j_b); n += 1
    if all(o >= 0 for o in obs):      # fully observed: local must equal exact exactly
        lw = max(lw, float(np.abs(local_predict(ch, obs, u, pc) - p_e).max()))
print(f"{n} random tiny sequences (u 6-9, <=5 missing, all tau, 3^missing completions): max |predictive diff| {worst_p:.2e}, max rel joint diff {worst_j:.2e}")
print(f"fully observed: max |local - exact| {lw:.2e}")
