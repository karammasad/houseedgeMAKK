"""When do real polarity flips happen? Change-point scan on TRAIN histories (analysis only).

  python -m method2_sizes.flip_time_probe --data data/main

For every long train machine: take its best XOR pair on the full history (flip-robust: pick the pair with the
largest sum of absolute half-segment effects), then scan flip time tau (every 4 ticks) with a two-segment Bernoulli
model whose two cells swap polarity at tau (profile MLE rates). Keep machines whose best flip beats no-flip by
>= 8 nats. Reports the distribution of tau / cutoff (deciles) by train condition (conditions used here for
ANALYSIS only; the model never sees them), and what share of machines flip.
"""
import argparse, csv, itertools, math, os
from collections import defaultdict
from common.data import load_split, public_dir, wins


def ll(k, n):
    if n == 0 or k == 0 or k == n:
        return 0.0
    p = k / n; return k * math.log(p) + (n - k) * math.log(1 - p)


def scan(w, cut):
    best = None
    for a, b in itertools.combinations(range(1, 9), 2):
        pts = [(t, w[t - a] ^ w[t - b], w[t]) for t in range(b, cut) if w[t] >= 0 and w[t - a] >= 0 and w[t - b] >= 0]
        if len(pts) < 120:
            continue
        # cumulative counts per cell
        n = [[0], [0]]; k = [[0], [0]]
        for _, x, y in pts:
            for c in (0, 1):
                n[c].append(n[c][-1] + (x == c)); k[c].append(k[c][-1] + (x == c and y == 1))
        N = [n[0][-1], n[1][-1]]; K = [k[0][-1], k[1][-1]]
        base = ll(K[0], N[0]) + ll(K[1], N[1])
        top = None
        for i in range(20, len(pts) - 20, 3):
            # before i keep cell x; from i on cells swap: cell A = (x=0 before) + (x=1 after)
            nA = n[0][i] + (N[1] - n[1][i]); kA = k[0][i] + (K[1] - k[1][i])
            nB = n[1][i] + (N[0] - n[0][i]); kB = k[1][i] + (K[0] - k[0][i])
            v = ll(kA, nA) + ll(kB, nB)
            if top is None or v > top[0]:
                top = (v, pts[i][0])
        gain = top[0] - base
        score = max(base, top[0])
        if best is None or score > best[0]:
            best = (score, (a, b), gain, top[1], base)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); q, seq = load_split(pub, "train")
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "train", "conditions.csv")))}
    res = defaultdict(list); n_tot = defaultdict(int)
    for r in q:
        c = cond[r["query_id"]]
        if c == "short_history":
            continue
        w = wins(seq[r["machine_id"]]); n_tot[c] += 1
        b = scan(w, r["cutoff"])
        if b and b[2] >= 8 and b[4] > -10 ** 9:
            res[c].append(b[3] / r["cutoff"])
    print("share of long train machines with a detected polarity flip (>= 8 nats) and where it happens:")
    for c in ("stable", "drift", "shared_change"):
        v = sorted(res[c]); n = len(v)
        dec = [0] * 10
        for x in v:
            dec[min(9, int(x * 10))] += 1
        print(f"  {c:14s} flips {n}/{n_tot[c]} ({n / max(1, n_tot[c]):.0%});  tau/cutoff histogram (deciles 0-10% ... 90-100%): {dec}")
        if n:
            print(f"                 quantiles tau/cutoff: q10 {v[n // 10]:.2f}  q25 {v[n // 4]:.2f}  median {v[n // 2]:.2f}  "
                  f"q75 {v[3 * n // 4]:.2f}  q90 {v[9 * n // 10]:.2f}")


if __name__ == "__main__":
    main()
