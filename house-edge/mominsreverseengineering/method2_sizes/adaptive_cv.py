"""Experiment 10 check: episode-grouped 5-fold CV of the feedback gates on TRAIN examples only.

  python -m method2_sizes.adaptive_cv --data data/main

Examples: each train query's label + self-supervised examples at cutoff-1 and cutoff-2. Every feature and P(win)
is computed from seq[:cutoff] only (asserted below). Folds group whole episodes.
Caveat: adi's and method 2's population priors are fitted on the same train histories (in-sample priors), so all
gates share that optimism equally; the comparison BETWEEN gates is what matters.
"""
import argparse, csv, os
from common.data import load_split, public_dir
from method2_sizes.round2 import FINAL
from method2_sizes.model import Model
from method2_sizes import adaptive as A


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); m = Model(**FINAL).fit(pub); tq, ts = load_split(pub, "train")
    # leakage guard: every example's history must end strictly before its cutoff (load_split builds seq of
    # length cutoff; truncation below keeps that invariant)
    assert all(len(ts[r["machine_id"]]) == r["cutoff"] for r in tq)
    with open(os.path.join(pub, "train", "labels.csv")) as f:
        lab = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(f)}
    ex = A.training_examples(m, tq, ts, labels=lab)
    res, thetas = A.episode_cv(ex)
    print(f"{len(ex)} examples from {len({e[3] for e in ex})} train episodes, 5 episode-grouped folds")
    print(f"{'gate':10s} {'OOF log-loss':>13s} {'OOF Brier':>10s}   vs short-log rule")
    for g, (ll, br) in res.items():
        print(f"{g:10s} {ll:13.5f} {br:10.5f}   log-loss {ll - res['short'][0]:+.5f}  Brier {br - res['short'][1]:+.5f}")
    print("learned theta per fold (a, b*H, c*short):", [[round(x, 2) for x in t] for t in thetas])


if __name__ == "__main__":
    main()
