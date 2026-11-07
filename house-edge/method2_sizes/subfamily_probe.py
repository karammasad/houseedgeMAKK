"""Are there latent machine subfamilies? Out-of-sample test of the machine-level win-rate distribution.

  python -m method2_sizes.subfamily_probe --data data/main

For every long TRAIN machine, use only the first 55% of its history (before any flip in every condition), find its
best XOR pair there, and record the high-cell and low-cell counts (k wins of n ticks). Per machine type, compare how
well three population models predict HELD-OUT machines' counts (5-fold CV grouped by whole episodes, log-likelihood
per machine):
  BB     one Beta prior (what every gate in the team uses)
  MIX2   a 2-component mixture of Betas (distinct subfamilies)
  NPMLE  a free discrete distribution on a 49-point grid (any shape), lightly smoothed
If NPMLE/MIX2 do not beat BB out of sample, the smooth single prior is adequate and clusters are not supported.
Reports the paired fold-level difference and its standard error, plus the NPMLE mass profile.
"""
import argparse, itertools, math
from collections import defaultdict
from common.data import load_split, public_dir, wins

GRID = [(i + 1) / 50 for i in range(49)]


def lbinom(k, n, p):
    return k * math.log(p) + (n - k) * math.log(1 - p)


def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def bb_ll(k, n, m, c):
    a, b = m * c, (1 - m) * c; return lbeta(a + k, b + n - k) - lbeta(a, b)


def lse(v):
    mx = max(v); return mx + math.log(sum(math.exp(x - mx) for x in v))


def machine_cells(w, cut):
    seg = int(.55 * cut); best = None
    for a, b in itertools.combinations(range(1, 9), 2):
        n = [0, 0]; k = [0, 0]
        for t in range(b, seg):
            if w[t] >= 0 and w[t - a] >= 0 and w[t - b] >= 0:
                x = w[t - a] ^ w[t - b]; n[x] += 1; k[x] += w[t]
        if min(n) >= 10:
            e = abs(k[1] / n[1] - k[0] / n[0])
            if best is None or e > best[0]:
                hi = 1 if k[1] / n[1] > k[0] / n[0] else 0
                best = (e, (k[hi], n[hi]), (k[1 - hi], n[1 - hi]))
    return best


def fit_bb(data):
    best = None
    for m in [i / 40 for i in range(1, 40)]:
        for c in (3, 6, 12, 25, 50, 100, 200):
            v = sum(bb_ll(k, n, m, c) for k, n in data)
            if best is None or v > best[0]:
                best = (v, m, c)
    return best[1:]


def fit_npmle(data, iters=150, smooth=.5):
    w = [1 / len(GRID)] * len(GRID); L = [[lbinom(k, n, g) for g in GRID] for k, n in data]
    for _ in range(iters):
        acc = [0.0] * len(GRID)
        for row in L:
            lp = [x + math.log(max(1e-300, wi)) for x, wi in zip(row, w)]; z = lse(lp)
            for j, x in enumerate(lp):
                acc[j] += math.exp(x - z)
        raw = [(x + smooth / len(GRID)) / (len(data) + smooth) for x in acc]
        w = [.25 * raw[max(0, j - 1)] + .5 * raw[j] + .25 * raw[min(len(GRID) - 1, j + 1)] for j in range(len(GRID))]
        z = sum(w); w = [x / z for x in w]
    return w


def npmle_ll(k, n, w):
    return lse([lbinom(k, n, g) + math.log(max(1e-300, wi)) for g, wi in zip(GRID, w)])


def fit_mix2(data, iters=60):
    best = None
    for m1, m2 in ((.2, .7), (.4, .85), (.1, .5), (.5, .95)):
        pi, ms, cs = .5, [m1, m2], [30, 30]
        for _ in range(iters):
            r1 = []
            for k, n in data:
                l1 = math.log(pi) + bb_ll(k, n, ms[0], cs[0]); l2 = math.log(1 - pi) + bb_ll(k, n, ms[1], cs[1]); z = lse([l1, l2])
                r1.append(math.exp(l1 - z))
            pi = min(.95, max(.05, sum(r1) / len(r1)))
            for j, rr in enumerate((r1, [1 - x for x in r1])):
                sw = sum(rr) or 1e-9; ms[j] = min(.97, max(.03, sum(x * (k / n) for x, (k, n) in zip(rr, data)) / sw))
                cs[j] = min(200, max(3, cs[j]))
                bc = max((sum(x * bb_ll(k, n, ms[j], c) for x, (k, n) in zip(rr, data)), c) for c in (3, 6, 12, 25, 50, 100, 200))
                cs[j] = bc[1]
        v = sum(lse([math.log(pi) + bb_ll(k, n, ms[0], cs[0]), math.log(1 - pi) + bb_ll(k, n, ms[1], cs[1])]) for k, n in data)
        if best is None or v > best[0]:
            best = (v, pi, tuple(ms), tuple(cs))
    return best[1:]


def mix2_ll(k, n, p):
    pi, ms, cs = p
    return lse([math.log(pi) + bb_ll(k, n, ms[0], cs[0]), math.log(1 - pi) + bb_ll(k, n, ms[1], cs[1])])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); q, seq = load_split(pub, "train")
    rows = defaultdict(list)
    for r in q:
        w = wins(seq[r["machine_id"]])
        if sum(v >= 0 for v in w) < 150:
            continue
        b = machine_cells(w, r["cutoff"])
        if b:
            rows[r["machine_type"]].append((r["episode_id"], b[1], b[2]))
    for cell, idx in (("HIGH cell", 1), ("LOW cell", 2)):
        print(f"\n=== {cell}: held-out log-likelihood per machine, 5-fold CV grouped by episode ===")
        for typ in ("steady", "balanced", "jackpot"):
            R = rows[typ]; eps = sorted({e for e, _, _ in R}); fold = {e: i % 5 for i, e in enumerate(eps)}
            tot = {"BB": [], "MIX2": [], "NPMLE": []}
            for f in range(5):
                tr = [x[idx] for x in R if fold[x[0]] != f]; te = [x[idx] for x in R if fold[x[0]] == f]
                if len(te) < 5:
                    continue
                m, c = fit_bb(tr); mx = fit_mix2(tr); wn = fit_npmle(tr)
                tot["BB"].append(sum(bb_ll(k, n, m, c) for k, n in te) / len(te))
                tot["MIX2"].append(sum(mix2_ll(k, n, mx) for k, n in te) / len(te))
                tot["NPMLE"].append(sum(npmle_ll(k, n, wn) for k, n in te) / len(te))
            def pd(name):
                d = [x - y for x, y in zip(tot[name], tot["BB"])]; mu = sum(d) / len(d)
                se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d)); return mu, se
            print(f"  {typ:9s} n={len(R):3d}  BB {sum(tot['BB']) / 5:8.3f}   MIX2 vs BB {pd('MIX2')[0]:+.3f} (se {pd('MIX2')[1]:.3f})   "
                  f"NPMLE vs BB {pd('NPMLE')[0]:+.3f} (se {pd('NPMLE')[1]:.3f})")
            if idx == 1:
                wn = fit_npmle([x[idx] for x in R]); prof = [round(sum(wn[j] for j in range(i, min(49, i + 5))), 2) for i in range(0, 49, 5)]
                print(f"            NPMLE mass by rate decile-ish (0-.1,.1-.2,...): {prof}")


if __name__ == "__main__":
    main()
