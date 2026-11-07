"""Method 2 diagnostics (standard library only). Reads TRAIN histories only.

  python -m method2_sizes.analysis --data data/main

B1: does the size of a win depend on earlier payouts (lags 1..8, previous win's size)?
Also: does the size mix move over time (first vs second half), and do machines in an
episode move together (shared size shifts)? Results inform FINDINGS.md.
"""
import argparse, math, random
from collections import defaultdict
from common.data import load_split, public_dir, SIZES

SI = {v: i for i, v in enumerate(SIZES)}


def g_test(table):
    """table: {row_key: [counts over 4 sizes]} -> (G statistic, dof). Rows with <10 obs dropped."""
    rows = [r for r in table.values() if sum(r) >= 10]
    if len(rows) < 2:
        return 0.0, 0
    col = [sum(r[j] for r in rows) for j in range(4)]; n = sum(col)
    g = 0.0
    for r in rows:
        nr = sum(r)
        for j in range(4):
            e = nr * col[j] / n
            if r[j] > 0 and e > 0:
                g += 2 * r[j] * math.log(r[j] / e)
    dof = (len(rows) - 1) * (sum(1 for c in col if c > 0) - 1)
    return g, dof


def cat(v):
    return "miss" if v < 0 else str(v)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data)
    q, seq = load_split(pub, "train")
    cond = {}
    import csv, os
    with open(os.path.join(pub, "train", "conditions.csv")) as f:
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}

    print("== B1: size of a win vs lagged payout (G-test of independence, per machine type) ==")
    print("   (pooled across machines; machine-level heterogeneity inflates G a bit, so compare to the")
    print("    within-machine shuffled null (50 draws); p = permutation p-value, which keeps each machine's own size mix)")
    rng = random.Random(0)
    for typ in ("steady", "balanced", "jackpot"):
        ms = [r for r in q if r["machine_type"] == typ]
        line = []
        for k in list(range(1, 9)) + ["prevwin"]:
            tab = defaultdict(lambda: [0] * 4); allobs = []
            for r in ms:
                s = seq[r["machine_id"]]
                obs = []
                last = None
                for t in range(len(s)):
                    v = s[t]
                    if v > 0:
                        if k == "prevwin":
                            key = cat(last) if last is not None else "none"
                        else:
                            key = cat(s[t - k]) if t - k >= 0 else "miss"
                        obs.append((key, SI[v]))
                    if v > 0:
                        last = v
                    elif v < 0:
                        pass
                for key, j in obs:
                    tab[key][j] += 1
                allobs.append(obs)
            g, d = g_test(tab)
            # null: shuffle sizes within each machine (keeps each machine's own size mix), 50 draws
            nulls = []
            for _ in range(50):
                null = defaultdict(lambda: [0] * 4)
                for obs in allobs:
                    sizes = [j for _, j in obs]; rng.shuffle(sizes)
                    for (key, _), j in zip(obs, sizes):
                        null[key][j] += 1
                nulls.append(g_test(null)[0])
            pval = (1 + sum(x >= g for x in nulls)) / 51
            line.append(f"{str(k):7s} G={g:5.1f}  null mean {sum(nulls)/50:5.1f}  p={pval:.2f}")
        print(f" {typ}:"); [print("   ", x) for x in line]

    print("\n== B1b: per-machine test (catches effects whose direction differs by machine) ==")
    print("   sum over machines of each machine's own G(size x lag-k win/loss) vs within-machine shuffle")
    for k in range(1, 9):
        def total_g(shuffle):
            tot = 0.0
            for r in q:
                s = seq[r["machine_id"]]
                obs = [(int(s[t - k] > 0), SI[s[t]]) for t in range(k, len(s)) if s[t] > 0 and s[t - k] >= 0]
                if shuffle:
                    js = [j for _, j in obs]; rng.shuffle(js); obs = [(x, j) for (x, _), j in zip(obs, js)]
                tab = defaultdict(lambda: [0] * 4)
                for x, j in obs:
                    tab[x][j] += 1
                tot += g_test(tab)[0]
            return tot
        g = total_g(False); nulls = [total_g(True) for _ in range(20)]
        pval = (1 + sum(x >= g for x in nulls)) / 21
        print(f"    lag {k}: G={g:7.1f}  null mean {sum(nulls) / 20:7.1f}  p={pval:.2f}")

    print("\n== Size mix over time: P(1x|win) first half vs second half of each machine, by condition ==")
    for c in ("stable", "short_history", "drift", "shared_change"):
        diffs = []
        for r in q:
            if cond[r["query_id"]] != c:
                continue
            s = [v for v in seq[r["machine_id"]] if v >= 0]
            h = len(s) // 2
            w1 = [v for v in s[:h] if v > 0]; w2 = [v for v in s[h:] if v > 0]
            if len(w1) >= 15 and len(w2) >= 15:
                p1 = sum(v == 1 for v in w1) / len(w1); p2 = sum(v == 1 for v in w2) / len(w2)
                # expected sd under no change
                p = (p1 * len(w1) + p2 * len(w2)) / (len(w1) + len(w2))
                sd = math.sqrt(max(p * (1 - p), 1e-9) * (1 / len(w1) + 1 / len(w2)))
                diffs.append((p2 - p1) / sd)
        if diffs:
            var = sum(d * d for d in diffs) / len(diffs)
            print(f"  {c:14s} n={len(diffs):3d}  var(z)={var:.2f}  (1.0 = no change; >1 = sizes move)")

    print("\n== Shared size shifts: within-episode correlation of machines' (late - early) P(big|win) ==")
    for c in ("stable", "drift", "shared_change"):
        eps = defaultdict(list)
        for r in q:
            if cond[r["query_id"]] != c:
                continue
            s = [v for v in seq[r["machine_id"]] if v >= 0]; h = len(s) // 2
            w1 = [v for v in s[:h] if v > 0]; w2 = [v for v in s[h:] if v > 0]
            if len(w1) >= 10 and len(w2) >= 10:
                big = lambda w: sum(v >= 2 for v in w) / len(w)
                eps[r["episode_id"]].append(big(w2) - big(w1))
        # intraclass: mean of pairwise products / variance
        allv = [x for v in eps.values() for x in v]; mu = sum(allv) / len(allv)
        var = sum((x - mu) ** 2 for x in allv) / len(allv)
        num = den = 0.0
        for v in eps.values():
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    num += (v[i] - mu) * (v[j] - mu); den += 1
        print(f"  {c:14s} ICC={num / den / var:+.2f}  mean shift={mu:+.3f}")


if __name__ == "__main__":
    main()
