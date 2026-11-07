"""Method 5, Phase 1: is the real data trustworthy as supervision, and what does it offer? (read-only, stdlib)

  python -m method5_simulation.data_checks --data data/main

Uses only histories and TRAIN labels (never validation/test labels or targets). Checks:
  1. split disjointness (episode and machine ids)             4. self-supervised example counts per split
  2. strict cutoff: every history tick < cutoff, label tick absent  5. one-hot target noise floor (train outcomes)
  3. short-history layout: 40 consecutive + 24 sampled older ticks; missing ticks stay unknown (-1), never 0
"""
import argparse, csv, os
from collections import Counter, defaultdict
from common.data import load_split, public_dir

SPLITS = ("train", "validation", "test")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); D = {s: load_split(pub, s) for s in SPLITS}
    print("1. split disjointness")
    for x in SPLITS:
        for y in SPLITS:
            if x < y:
                ep = {r["episode_id"] for r in D[x][0]} & {r["episode_id"] for r in D[y][0]}
                mc = {r["machine_id"] for r in D[x][0]} & {r["machine_id"] for r in D[y][0]}
                print(f"   {x} vs {y}: shared episodes {len(ep)}, shared machines {len(mc)}")
    print("2. strict cutoff and 3. short-history layout")
    for s in SPLITS:
        q, seq = D[s]; bad = 0; kinds = Counter(); vals = Counter(); ok_layout = 0; n_short = 0
        for r in q:
            x = seq[r["machine_id"]]; cut = r["cutoff"]
            bad += len(x) != cut                       # load_split builds seq of exactly `cutoff` ticks (< cutoff)
            vals.update(v for v in x)
            obs = [t for t, v in enumerate(x) if v >= 0]
            if len(obs) < cut * .6:
                n_short += 1
                block = all(x[t] >= 0 for t in range(cut - 40, cut)); older = [t for t in obs if t < cut - 40]
                ok_layout += block and len(older) == 24
            kinds["short" if len(obs) < cut * .6 else "long"] += 1
        print(f"   {s}: {len(q)} machines ({dict(kinds)}); history length != cutoff: {bad}; short layout 40+24 ok: {ok_layout}/{n_short}; "
              f"values seen: {sorted(vals)}")
    # label tick is not in any history: the train label is the payout AT the cutoff, which load_split never loads
    lab = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(open(os.path.join(pub, "train", "labels.csv")))}
    print("   train labels read:", len(lab), "values", sorted(set(lab.values())))
    print("4. self-supervised next-tick examples available (an observed tick t with all 8 predecessors observed)")
    tot = 0
    for s in SPLITS:
        q, seq = D[s]; n = 0; m = 0
        for r in q:
            x = seq[r["machine_id"]]
            for t in range(8, len(x)):
                if x[t] >= 0:
                    m += 1; n += all(x[t - l] >= 0 for l in range(1, 9))
        tot += n; print(f"   {s}: {n} examples with full 8-lag context ({m} observed ticks with any context)")
    print(f"   total usable (all splits' HISTORIES; no labels needed): {tot}")
    print("5. one-hot target noise floor (train outcomes by machine type; mean of 1 - sum_k p_k^2)")
    q, seq = D["train"]; by = defaultdict(Counter)
    for r in q:
        by[r["machine_type"]].update(v for v in seq[r["machine_id"]] if v >= 0)
    for t, c in sorted(by.items()):
        n = sum(c.values()); p = [c[v] / n for v in (0, 1, 2, 5, 20)]
        print(f"   {t:9s} mix {[round(x, 3) for x in p]}  noise floor if only the type mean is known: {1 - sum(x * x for x in p):.3f}")


if __name__ == "__main__":
    main()
