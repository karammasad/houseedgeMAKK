"""C0: why do XOR triples win per-machine discovery but not help the model? (adi / method 3)

For each long-history train machine:
  1. pick the best structure on the first 70% of ticks (singles, pairs, triples, lags 1..12, penalty),
  2. score the best pair and the best triple on the held-out last 30% (per-spin log-lik),
  3. measure how often the best triple's XOR equals the best pair's XOR (aliasing),
  4. check whether the triple is "pair-substituted": triple = (pair lag) + two lags whose XOR predicts the other pair lag.
Usage: python -m adi.c0_analysis --data data/main --split train
"""
import argparse, itertools as it, math
from collections import Counter
from common.data import public_dir, load_split, wins


def cells(y, g):
    c = {}
    for yy, gg in zip(y, g):
        c.setdefault(gg, [0, 0])[yy] += 1
    return c


def loglik(y, g, fit=None):
    """Log-lik of y given grouping g, with rates from `fit` cells (held-out) or in-sample."""
    c = fit if fit is not None else cells(y, g)
    tot = 0.0
    for yy, gg in zip(y, g):
        n0, n1 = c.get(gg, [0, 0]); p = (n1 + .5) / (n0 + n1 + 1)
        tot += math.log(p if yy else 1 - p)
    return tot


def xor(w, t, s):
    x = 0
    for l in s:
        x ^= w[t - l]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/main"); ap.add_argument("--split", default="train")
    ap.add_argument("--max-lag", type=int, default=12); ap.add_argument("--penalty", type=float, default=2.0)
    a = ap.parse_args()
    q, seq = load_split(public_dir(a.data), a.split)
    L = a.max_lag
    pairs = list(it.combinations(range(1, L + 1), 2)); triples = list(it.combinations(range(1, L + 1), 3))
    singles = [(l,) for l in range(1, L + 1)]
    stats = Counter(); agree = []; ho = {"pair": 0.0, "triple": 0.0, "n": 0}
    subst = Counter(); trip_wins_ho = 0; nmach = 0
    for r in q:
        w = wins(seq[r["machine_id"]])
        if sum(x < 0 for x in w) > len(w) / 2:
            continue
        T = [t for t in range(L, len(w)) if w[t] >= 0 and all(w[t - l] >= 0 for l in range(1, L + 1))]
        if len(T) < 60:
            continue
        nmach += 1
        cut = int(.7 * len(T)); Tf, Th = T[:cut], T[cut:]
        yf = [w[t] for t in Tf]; yh = [w[t] for t in Th]
        def best(cands):
            top = None
            for s in cands:
                g = [xor(w, t, s) for t in Tf]
                ll = loglik(yf, g) - a.penalty * (len(s) - 1)
                if top is None or ll > top[0]:
                    top = (ll, s)
            return top
        bs, bp, bt = best(singles), best(pairs), best(triples)
        overall = max((bs, bp, bt), key=lambda x: x[0])
        stats[len(overall[1])] += 1
        # held-out log-lik of best pair vs best triple, rates fitted on the first 70%
        for name, s in (("pair", bp[1]), ("triple", bt[1])):
            gf = [xor(w, t, s) for t in Tf]; gh = [xor(w, t, s) for t in Th]
            ho[name] += loglik(yh, gh, cells(yf, gf))
        ho["n"] += len(Th)
        if len(overall[1]) == 3:
            gf_p = [xor(w, t, bp[1]) for t in Tf]; gh_p = [xor(w, t, bp[1]) for t in Th]
            gf_t = [xor(w, t, bt[1]) for t in Tf]; gh_t = [xor(w, t, bt[1]) for t in Th]
            if loglik(yh, gh_t, cells(yf, gf_t)) > loglik(yh, gh_p, cells(yf, gf_p)):
                trip_wins_ho += 1
            agree.append(sum(xor(w, t, bp[1]) == xor(w, t, bt[1]) for t in T) / len(T))
            # aliasing: triple = one pair lag l + (j,k), where wins at t-l are themselves gated by XOR at lags (j-l, k-l)
            P, Tr = set(bp[1]), set(bt[1])
            shared = P & Tr
            subst["shares a lag with best pair" if shared else "no shared lag"] += 1
            if len(shared) == 1:
                (keep,) = shared; (other,) = P - shared; rest = tuple(sorted(Tr - shared))
                rel = tuple(sorted(x - other for x in rest))
                subst["rest - other_pair_lag == best pair (alias)" if rel == tuple(sorted(bp[1])) else "shared, not alias"] += 1
    print(f"{nmach} long-history machines, split={a.split}, lags 1..{L}, penalty {a.penalty}")
    print("best structure order (fit on first 70%):", dict(stats))
    print(f"held-out per-spin log-lik: best pair {ho['pair'] / ho['n']:.4f}   best triple {ho['triple'] / ho['n']:.4f}")
    if agree:
        agree.sort()
        print(f"machines where a triple won: {len(agree)}; triple beats pair on held-out ticks in {trip_wins_ho}")
        print("agreement of triple-XOR with pair-XOR, quartiles:", [round(agree[int(p * (len(agree) - 1))], 3) for p in (.25, .5, .75)])
        for k, v in subst.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
