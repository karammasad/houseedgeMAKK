"""Structure discovery: which lags gate wins? Owner: Person C. Standard library only.

For every machine, scores each candidate structure (single lag, XOR pair, XOR triple up to --max-lag)
by how well XOR(wins at those lags) or its negation predicts the next win (best of the two polarities),
then prints how often each structure is the best one and its mean per-spin log-likelihood gain.
Usage: python -m method3_discovery.discover --data data/main --split train --max-lag 12 --max-order 3
"""
import argparse, itertools as it, math
from collections import Counter
from common.data import public_dir, load_split, wins


def loglik(y, groups):
    tot = 0.0
    cells = {}
    for yy, g in zip(y, groups):
        cells.setdefault(g, [0, 0])[yy] += 1
    for n0, n1 in cells.values():
        n = n0 + n1; p = (n1 + .5) / (n + 1)
        tot += n1 * math.log(p) + n0 * math.log(1 - p)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/main"); ap.add_argument("--split", default="train")
    ap.add_argument("--max-lag", type=int, default=12); ap.add_argument("--max-order", type=int, default=3)
    ap.add_argument("--penalty", type=float, default=2.0, help="extra log-lik each extra lag must earn")
    a = ap.parse_args()
    q, seq = load_split(public_dir(a.data), a.split)
    cands = [s for k in range(1, a.max_order + 1) for s in it.combinations(range(1, a.max_lag + 1), k)]
    best = Counter(); gains = []
    for r in q:
        w = wins(seq[r["machine_id"]])
        if sum(x < 0 for x in w) > len(w) / 2:
            continue  # short histories: too few complete windows to judge
        t0 = a.max_lag
        T = [t for t in range(t0, len(w)) if w[t] >= 0 and all(w[t - l] >= 0 for l in range(1, a.max_lag + 1))]
        y = [w[t] for t in T]; base = loglik(y, [0] * len(y))
        top = (base, ("none",))
        for s in cands:
            g = [0] * len(T)
            for i, t in enumerate(T):
                x = 0
                for l in s:
                    x ^= w[t - l]
                g[i] = x
            ll = loglik(y, g) - a.penalty * (len(s) - 1)
            if ll > top[0]:
                top = (ll, s)
        best[top[1]] += 1; gains.append((top[0] - base) / max(len(T), 1))
    n = sum(best.values())
    print(f"{n} machines with long histories, split={a.split}, lags 1..{a.max_lag}, order <= {a.max_order}")
    gains.sort(); print("per-spin log-lik gain quartiles:", [round(gains[int(p * (len(gains) - 1))], 3) for p in (.25, .5, .75)])
    for s, c in best.most_common(15):
        print(f"  {str(s):14s} {c:5d}  ({c / n:.1%})")


if __name__ == "__main__":
    main()
