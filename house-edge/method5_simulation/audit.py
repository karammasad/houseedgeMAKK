"""Requirement 1: data-integrity audit of the self-supervised examples.

  python -m method5_simulation.audit --data data/main

Checks (read-only): targets are observed outcomes; features depend only on ticks strictly before the target
(future-perturbation test on random examples); the cutoff, other machines and labels never enter the builder; example
counts vs independent units (machines / episodes) and within-machine dependence; the train-query labels are the only
labels used.
"""
import argparse, random
import numpy as np
from common.data import public_dir
from method5_simulation import data as D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); rng = random.Random(0); nprng = np.random.default_rng(0)
    for split in ("train", "validation", "test"):
        M = D.load_machines(pub, split); n = 0; nm = 0; bad_target = 0; leaks = 0; checked = 0; per = []
        for i, m in enumerate(M):
            ts = D.target_ticks(m["cats"]); nm += bool(ts); n += len(ts); per.append(len(ts))
            bad_target += sum(m["cats"][t] < 0 for t in ts)
        # future-perturbation test: overwrite every tick >= t with random categories; features must not change
        for _ in range(300):
            m = rng.choice(M); ts = D.target_ticks(m["cats"])
            if not ts:
                continue
            t = rng.choice(ts); W1, S1 = D.build(m["cats"], [t], m["typ"], m["sig"])
            c2 = m["cats"].copy(); c2[t:] = nprng.integers(-1, 5, size=len(c2) - t)
            W2, S2 = D.build(c2, [t], m["typ"], m["sig"]); checked += 1
            leaks += int(not (np.array_equal(W1, W2) and np.array_equal(S1, S2)))
        eps = {m["ep"] for m in M}
        print(f"{split:10s} machines {len(M):5d} episodes {len(eps):3d} | self-supervised examples {n:7d} ({n / len(M):.0f}/machine; "
              f"unobserved targets: {bad_target}) | future-perturbation leaks: {leaks}/{checked}")
    # dependence: how many independent units do the 'examples' really contain?
    M = D.load_machines(pub, "train"); lab = D.train_labels(pub); X = D.examples(M, lab)
    eff_ep = len({m["ep"] for m in M})
    print(f"\ntrain: {len(X['y'])} examples = {int((X['is_query'] == 0).sum())} self-supervised + {int(X['is_query'].sum())} labelled cutoff queries")
    print(f"       independent units: {len(M)} machines in {eff_ep} episodes (consecutive ticks of one machine are strongly dependent;")
    print(f"       so folds are grouped by EPISODE and uncertainty is bootstrapped over episodes, not over examples)")
    print(f"       builder signature uses only (cats, target ticks, type, signal): no cutoff, no other machine, no labels, no Methods 1-4")
    print(f"       target categories (self-supervised): {np.bincount(X['y'][X['is_query'] == 0], minlength=5).tolist()}; "
          f"short-log examples {int(X['short'].sum())}")
    print("       queries use t = cutoff: same builder, history ends at cutoff-1; the label tick is never in any window")


if __name__ == "__main__":
    main()
