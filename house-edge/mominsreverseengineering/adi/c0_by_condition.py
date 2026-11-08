"""C0 follow-up: held-out pair vs triple log-lik per condition (train labels only; conditions read for analysis, never in a model)."""
import argparse, csv, os, itertools as it
from collections import defaultdict
from common.data import public_dir, load_split, wins
from adi.c0_analysis import cells, loglik, xor


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--split", default="train")
    a = ap.parse_args(); pub = public_dir(a.data); L = 12
    q, seq = load_split(pub, a.split)
    with open(os.path.join(pub, a.split, "conditions.csv")) as f:
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}
    pairs = list(it.combinations(range(1, L + 1), 2)); triples = list(it.combinations(range(1, L + 1), 3))
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # pair, triple, pair-in-sample-on-heldout-window, n
    for r in q:
        w = wins(seq[r["machine_id"]])
        if sum(x < 0 for x in w) > len(w) / 2:
            continue
        T = [t for t in range(L, len(w)) if w[t] >= 0 and all(w[t - l] >= 0 for l in range(1, L + 1))]
        cut = int(.7 * len(T)); Tf, Th = T[:cut], T[cut:]
        yf = [w[t] for t in Tf]; yh = [w[t] for t in Th]
        bp = max(pairs, key=lambda s: loglik(yf, [xor(w, t, s) for t in Tf]))
        bt = max(triples, key=lambda s: loglik(yf, [xor(w, t, s) for t in Tf]))
        c = acc[cond[r["query_id"]]]
        for i, s in enumerate((bp, bt)):
            c[i] += loglik(yh, [xor(w, t, s) for t in Th], cells(yf, [xor(w, t, s) for t in Tf]))
        c[2] += loglik(yh, [xor(w, t, bp) for t in Th])  # pair refit on the held-out window itself (oracle polarity)
        c[3] += len(Th)
    print(f"{'condition':14s} {'pair':>8s} {'triple':>8s} {'pair(refit)':>12s}")
    for k, (p, t, o, n) in sorted(acc.items()):
        print(f"{k:14s} {p / n:8.3f} {t / n:8.3f} {o / n:12.3f}")


if __name__ == "__main__":
    main()
