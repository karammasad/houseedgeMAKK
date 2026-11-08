"""Cheap falsification: does the payout SIZE given a win depend on anything Thanos leaves out? Thanos: size | win ~ V_T[b_t xor d], b_t = [payout at lag l >= 2],
independent of the gate side, polarity, the previous outcomes and the finer payout category. TRAIN histories only, 576 long machines (production top pair / flip time
for stratification; both use the machine's whole history, which is legitimate here because the test conditions on nothing that depends on the size labels being tested).
Permutation test: within each machine x b stratum, shuffle the candidate covariate among the machine's wins; statistic = sum over machines/strata of the G statistic
(2 x 4 tables, size category vs covariate). 400 permutations; p = share >= observed. Machine-level effect size also reported (mean |TV distance|)."""
import pickle, math, sys
import numpy as np
from common.data import public_dir, load_split
R = pickle.load(open("experiments/forensics/rows.pkl", "rb")); M = list({r["mid"]: r for r in R if r["kind"] == "pseudo"}.values())
q, seq = load_split(public_dir("../data/main"), "train"); SZ = {1: 0, 2: 1, 5: 2, 20: 3}; rng = np.random.default_rng(0)
def G(tab):
    tab = tab[:, tab.sum(0) > 0]
    if tab.shape[1] < 2 or tab.sum() < 12: return 0.0
    e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum(); m = tab > 0; return float(2 * (tab[m] * np.log(tab[m] / e[m])).sum())
def build(m):
    s = seq[m["mid"]]; i, j = m["pair"]; k = min(i, j); L = len(s); w = [int(v > 0) if v >= 0 else -1 for v in s]; tau = m["tau"]
    T = [t for t in range(8, L) if w[t] == 1 and all(w[t - l] >= 0 for l in (i, j, 1, 2)) and s[t - k] >= 0]
    gate = {t: (w[t - i] ^ w[t - j]) ^ (1 if (tau is not None and t >= tau) else 0) for t in T}
    allt = [t for t in range(8, L) if all(w[t - l] >= 0 for l in (i, j)) and w[t] >= 0]
    rate = [np.mean([w[t] for t in allt if ((w[t - i] ^ w[t - j]) ^ (1 if (tau is not None and t >= tau) else 0)) == g]) if any(((w[t - i] ^ w[t - j]) ^ (1 if (tau is not None and t >= tau) else 0)) == g for t in allt) else 0 for g in (0, 1)]
    on = int(rate[1] > rate[0])
    rows = []
    for t in T:
        rows.append(dict(c=SZ[s[t]], b=int(s[t - k] >= 2), side=int(gate[t] == on), w1=w[t - 1], w2=w[t - 2], fine=(0 if s[t - k] < 2 else SZ[s[t - k]] - 0), half=int(t >= L // 2)))
    return rows
covs = {"gate side (on/off) | b": "side", "win at lag 1 | b": "w1", "win at lag 2 | b": "w2", "log half (early/late) | b": "half"}
data = [(m, build(m)) for m in M]; print(len(data), "machines;", sum(len(r) for _, r in data), "wins")
def stat(rows, cov, perm=False, fine=False):
    tot = 0.0
    for b in (0, 1):
        rr = [r for r in rows if r["b"] == b and (not fine or b == 1)]
        if len(rr) < 12: continue
        x = np.array([r[cov] for r in rr]); c = np.array([r["c"] for r in rr])
        if perm: x = rng.permutation(x)
        tab = np.zeros((int(x.max()) + 1, 4)); np.add.at(tab, (x, c), 1); tot += G(tab)
    return tot
for nm, cov in list(covs.items()) + [("finer lag-k payout 2/5/20 | b=1", "fine")]:
    fine = cov == "fine"
    obs = sum(stat(r, cov, fine=fine) for _, r in data); nul = []
    for _ in range(300): nul.append(sum(stat(r, cov, perm=True, fine=fine) for _, r in data))
    p = (1 + sum(n >= obs for n in nul)) / 301; print(f"{nm:34s} G obs {obs:8.1f}  null mean {np.mean(nul):8.1f} sd {np.std(nul):5.1f}  p = {p:.3f}")
# effect size for gate side: pooled size distribution by side within b, per type
print("\nsize distribution given a win, by gate side (on/off) within b, pooled per type (share of 1x / 2x / 5x / 20x):")
for ty in ("steady", "balanced", "jackpot"):
    for b in (0, 1):
        out = []
        for side in (1, 0):
            c = np.zeros(4)
            for m, rows in data:
                if m["type"] != ty: continue
                for r in rows:
                    if r["b"] == b and r["side"] == side: c[r["c"]] += 1
            out.append((c.sum(), np.round(c / c.sum(), 3)))
        print(f"  {ty:9s} b={b}: on n={int(out[0][0]):5d} {out[0][1]} | off n={int(out[1][0]):5d} {out[1][1]}")
