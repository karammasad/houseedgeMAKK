"""Aggregate the 5 CV folds. PRIMARY metric = held-out labelled cutoff queries; late-tick Brier is a training diagnostic.

  python -m method5_simulation.report

Brier = mean sum_k (p_k - y_k)^2 on REAL observed next outcomes: the competition's squared error to the true
probabilities plus a noise constant shared by every model, so differences between models are what matter.
Paired differences and 95% CIs are bootstrapped over EPISODES (examples within a machine/episode are dependent).
Models present in the results are reported; S3b (mask-preserving order shuffle) is included if cv_ablate has run.
"""
import glob
import numpy as np

NAMES = {"B0": "type x signal prior", "B1": "boosting, window frequencies (no order)", "B2": "boosting + last 8 outcomes",
         "S1": "GRU dense (order + gaps)", "S2": "GRU observed-only (no gaps)",
         "S3": "GRU columns shuffled (removes order AND gap positions)", "S3b": "GRU categories shuffled, mask fixed (removes order only)",
         "S1_seed1": "GRU dense, second training seed (noise control)",
         "S2_seed1": "GRU observed-only, second training seed"}


def load():
    folds = []
    for k in range(5):
        f = f"method5_simulation/results/cv_fold{k}.npz"
        try:
            z = dict(np.load(f, allow_pickle=True))
        except FileNotFoundError:
            continue
        for extra in (f"cv_ablate_fold{k}", f"cv_seed_fold{k}"):
            try:
                z.update(dict(np.load(f"method5_simulation/results/{extra}.npz", allow_pickle=True)))
            except FileNotFoundError:
                pass
        folds.append(z)
    return folds


def brier(P, y):
    return ((P - np.eye(5)[y]) ** 2).sum(1)


def main():
    folds = load(); keys = [k for k in NAMES if all(k in z for z in folds)]; rng = np.random.default_rng(0)
    cat = {k: np.concatenate([z[k] for z in folds]) for k in keys + ["y", "ep", "short", "is_query"]}
    y, ep = cat["y"], cat["ep"]; q = cat["is_query"].astype(bool); sh = cat["short"].astype(bool)
    print(f"{len(folds)} folds; {len(y)} held-out examples; {int(q.sum())} labelled cutoff queries (PRIMARY) from {len(np.unique(ep))} episodes\n")
    print("FOLD-BY-FOLD held-out cutoff-query Brier (primary):")
    print("  " + f"{'model':8s}" + "".join(f"{'fold ' + str(i):>10s}" for i in range(len(folds))) + f"{'pooled':>10s}")
    for k in keys:
        per = [brier(z[k][z['is_query'].astype(bool)], z['y'][z['is_query'].astype(bool)]).mean() for z in folds]
        print("  " + f"{k:8s}" + "".join(f"{v:10.5f}" for v in per) + f"{brier(cat[k][q], y[q]).mean():10.5f}")
    sel = [("PRIMARY: labelled cutoff queries", q), ("  of which short logs", q & sh), ("  of which long logs", q & ~sh),
           ("diagnostic: all late-tick examples", np.ones(len(y), bool)), ("diagnostic: late ticks, short logs", sh), ("diagnostic: late ticks, long logs", ~sh)]
    for label, m in sel:
        E = np.unique(ep[m]); per = {k: np.array([brier(cat[k][m & (ep == e)], y[m & (ep == e)]).mean() for e in E]) for k in keys}
        print(f"\n== {label} (n={int(m.sum())}, {len(E)} episodes) ==")
        print(f"  {'model':58s} {'Brier':>8s}   vs B0 [95% CI]                 vs B2 [95% CI]")
        for k in keys:
            def ci(a, b):
                d = a - b; bs = [rng.choice(d, len(d)).mean() for _ in range(1000)]; return d.mean(), np.percentile(bs, [2.5, 97.5])
            m0, c0 = ci(per[k], per["B0"]); m2, c2 = ci(per[k], per["B2"])
            print(f"  {k + ' ' + NAMES[k]:58s} {per[k].mean():8.5f}   {m0:+.5f} [{c0[0]:+.5f},{c0[1]:+.5f}]   {m2:+.5f} [{c2[0]:+.5f},{c2[1]:+.5f}]")
        if "S1" in per:
            for other in ("S3", "S3b", "S2", "S1_seed1"):
                if other in per:
                    d = per["S1"] - per[other]; bs = [rng.choice(d, len(d)).mean() for _ in range(1000)]
                    print(f"  S1 minus {other}: {d.mean():+.5f} [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]   (negative = S1 better)")


if __name__ == "__main__":
    main()
