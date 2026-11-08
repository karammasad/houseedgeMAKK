"""B1d: search for what drives the binary size state, on TRAIN histories only.
Outcome per observed win: is_1x. Candidate features (per machine, direction learned per machine):
  own lag k in {loss, 1x, big}; XNOR pairs of win/loss, of is-big, of is-1x; other machines' lag-k win/loss.
Score = sum over machines of per-machine log-likelihood gain (2x = G) vs a within-machine shuffle.
  python3 -m method2_sizes.state_search --data data/main
"""
import argparse, itertools, math, random
from collections import defaultdict
from common.data import load_split, public_dir


def cat3(v):
    return None if v < 0 else (0 if v == 0 else 1 if v == 1 else 2)


def g_binary(pairs):
    """pairs: list of (key, y in {0,1}) -> G statistic of y vs key."""
    tab = defaultdict(lambda: [0, 0])
    for k, y in pairs:
        if k is not None:
            tab[k][y] += 1
    n = sum(a + b for a, b in tab.values())
    if n == 0 or len(tab) < 2:
        return 0.0
    p = sum(b for _, b in tab.values()) / n
    if p in (0, 1):
        return 0.0
    g = 0.0
    for a, b in tab.values():
        m = a + b
        for c, e in ((a, m * (1 - p)), (b, m * p)):
            if c > 0:
                g += 2 * c * math.log(c / e)
    return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(); pub = public_dir(a.data)
    q, seq = load_split(pub, "train")
    eps = defaultdict(list)
    for r in q:
        eps[r["episode_id"]].append(r["machine_id"])
    F = {}
    for k in range(1, 9):
        F[f"own lag{k} loss/1x/big"] = lambda s, t, k=k, o=None: cat3(s[t - k])
    for a_, b_ in itertools.combinations(range(1, 9), 2):
        F[f"own xnor-win {a_},{b_}"] = lambda s, t, a_=a_, b_=b_, o=None: None if min(s[t - a_], s[t - b_]) < 0 else int((s[t - a_] > 0) == (s[t - b_] > 0))
        F[f"own xnor-big {a_},{b_}"] = lambda s, t, a_=a_, b_=b_, o=None: None if min(s[t - a_], s[t - b_]) < 0 else int((s[t - a_] > 1) == (s[t - b_] > 1))
        F[f"own xnor-1x {a_},{b_}"] = lambda s, t, a_=a_, b_=b_, o=None: None if min(s[t - a_], s[t - b_]) < 0 else int((s[t - a_] == 1) == (s[t - b_] == 1))
    for k in (1, 2, 3):
        F[f"other machines lag{k} win count"] = lambda s, t, k=k, o=None: None if any(x[t - k] < 0 for x in o) else sum(x[t - k] > 0 for x in o)
    rng = random.Random(0)
    tot = defaultdict(float); null = defaultdict(float)
    for r in q:
        s = seq[r["machine_id"]]
        others = [seq[m] for m in eps[r["episode_id"]] if m != r["machine_id"] and len(seq[m]) >= len(s)]
        ts = [t for t in range(8, len(s)) if s[t] > 0]
        ys = [int(s[t] == 1) for t in ts]; sh = ys[:]; rng.shuffle(sh)
        for name, f in F.items():
            keys = [f(s, t, o=others) for t in ts]
            tot[name] += g_binary(list(zip(keys, ys))); null[name] += g_binary(list(zip(keys, sh)))
    res = sorted(((tot[k] - null[k], k) for k in F), reverse=True)
    print("excess G over within-machine shuffle (larger = stronger per-machine dependence of is_1x):")
    for v, k in res[:a.top]:
        print(f"  {v:8.1f}  {k}   (G {tot[k]:.0f}, null {null[k]:.0f})")


if __name__ == "__main__":
    main()
