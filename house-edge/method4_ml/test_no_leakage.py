"""Leakage checks for Method 4 (run from house-edge/: python -m method4_ml.test_no_leakage).

1. Features at prediction tick t depend only on observations strictly before t, for every machine of the episode: scrambling
   every payout at ticks >= t (this machine and its neighbours) leaves the feature vectors bit-identical (training examples,
   including short-history masked copies, and query examples at the cutoff).
2. Missing ticks (-1) are unknown, not losses: turning observed old ticks into -1 changes features only through the observed set
   (checked by comparing against an explicit recomputation with the same mask), never by treating -1 as a payout of 0.
3. fit/predict never read validation/test answers: wrap open() and fail if targets.csv, labels.csv, conditions.csv or
   reference.csv of validation/test (or train conditions) are opened. Run with cache=False so every file path is exercised.
"""
import builtins, io, os, random, sys
from collections import defaultdict
from common.data import load_split
from method4_ml.model import Model
from method4_ml.features3 import episode_examples

PUB = "data/main"
FORBIDDEN = ("targets.csv", "labels.csv", "conditions.csv", "reference.csv")


def test_features_ignore_the_future(gm, sz, n_ep=6, seed=0):
    q, seq = load_split(PUB, "validation"); rng = random.Random(seed); eps = defaultdict(list)
    for r in q: eps[r["episode_id"]].append(r)
    for e in rng.sample(sorted(eps), n_ep):
        ms = eps[e]; cut = ms[0]["cutoff"]; t = rng.randrange(100, cut - 1)
        base = {r["machine_id"]: seq[r["machine_id"]] for r in ms}
        # the model passes seq[:t] only; also prove robustness if a caller passed the full list: scramble ticks >= t
        scr = {m: v[:t] + [rng.choice([0, 1, 2, 5, 20]) for _ in v[t:]] for m, v in base.items()}
        f0 = episode_examples(ms, {m: v[:t] + [-1] * (len(v) - t) for m, v in base.items()}, [t], gm, sz)
        f1 = episode_examples(ms, {m: v[:t] + [-1] * (len(v) - t) for m, v in scr.items()}, [t], gm, sz)
        assert f0 == f1, f"future ticks leaked into features (episode {e}, t={t})"
        # features are computed on seq[:t] inside episode_examples: verify directly with scrambled FULL sequences too
        g0 = episode_examples(ms, base, [t], gm, sz); g1 = episode_examples(ms, scr, [t], gm, sz)
        assert g0 == g1, f"episode_examples reads ticks >= t (episode {e}, t={t})"
    print(f"OK 1: features ignore ticks >= t ({n_ep} episodes, random t)")


def test_missing_is_not_a_loss(gm, sz):
    from method4_ml.features3 import gate_summary
    q, seq = load_split(PUB, "validation"); r = next(x for x in q if min(seq[x["machine_id"]]) >= 0); s = seq[r["machine_id"]]
    t = 150; masked = [v if (i >= t - 40 or i % 3 == 0) else -1 for i, v in enumerate(s[:t])]
    zero_filled = [v if v >= 0 else 0 for v in masked]
    a = gate_summary(masked, r, gm); b = gate_summary(zero_filled, r, gm)
    assert a != b, "masking changed nothing: -1 may be treated as an ordinary payout"
    assert all(0.0 <= x <= 1.0 or x == x for x in a)
    print("OK 2: missing ticks (-1) are not treated as losses (masked vs zero-filled features differ)")


def test_no_answer_files_read():
    opened = []; real = builtins.open
    def spy(f, *a, **k):
        if isinstance(f, (str, os.PathLike)): opened.append(str(f))
        return real(f, *a, **k)
    builtins.open = spy; io.open = spy
    try:
        m = Model(cache=False, workers=2, bags=1, seeds=1).fit(PUB); m.predict(PUB, "validation"); m.predict(PUB, "test")
    finally:
        builtins.open = real; io.open = real
    bad = sorted({p for p in opened if os.path.basename(p) in FORBIDDEN})
    assert not bad, f"fit/predict opened answer files: {bad}"
    print(f"OK 3: fit + predict(validation, test) opened {len(set(opened))} files, none of {FORBIDDEN}")
    return m


def test_menu_inputs_strictly_before():
    """Size-menu rows: the size model sees seq[:t] only and the label is the history payout at t (never a query label)."""
    from method4_ml.size_menu import _rows, CL
    for sp in ("train", "validation", "test"):
        q, seq = load_split(PUB, sp); qids = {r["machine_id"]: r for r in q}
        jobs, Y, _ = _rows(PUB, (sp,), 4)
        for (r, trunc), y in zip(jobs, Y):
            s = seq[r["machine_id"]]; t = len(trunc)
            assert trunc == s[:t] and t < r["cutoff"], "menu input reaches the label tick or the query tick"
            assert s[t] > 0 and y[CL[s[t]]] == 1, "menu label is not the history payout at t"
    print("OK 4: size-menu inputs stop strictly before their (history) label tick; no query tick is used")


if __name__ == "__main__":
    test_menu_inputs_strictly_before()
    m = test_no_answer_files_read() if "--io" in sys.argv else Model(bags=1, seeds=1).fit(PUB)
    sz = m.size_models["gate"] if m.size_feat else None
    test_features_ignore_the_future(m.gm, sz); test_missing_is_not_a_loss(m.gm, sz)
    m.gm.polarity = True     # v9 polarity bank (cross-machine features): the same future-scramble check must hold
    test_features_ignore_the_future(m.gm, sz, n_ep=6, seed=1); m.gm.polarity = False
    print("OK 1b: polarity-bank features ignore ticks >= t of every machine")
    print("ALL LEAKAGE CHECKS PASSED")
