"""Phase 1 diagnostic: how much could exact marginalization over unobserved ticks recover on short logs?

  python -m method2_sizes.missing_diag MODEL.pkl PARAMS.json --data data/main [--reps 4]

Uses the PPC-validated simulator (known true next-spin distribution) with the REAL short-log observation masks
(40 consecutive ticks + 24 sampled older ticks). Three views of the same simulated history, scored with the fitted
FINAL model (adi gate + method 2 sizes):
  A current   real mask; ticks with unobserved lags are skipped by the gate (their outcomes are discarded)
  B revealed  the 8 lags before each sampled older tick are revealed: an UPPER BOUND on anything exact latent-state
              marginalization of those ticks could recover (it knows the hidden bits; it also sees extra outcomes)
  C full      every tick observed (oracle for the total cost of the mask)
Reports win / size / total squared error vs truth, and the share of queries whose forecast moves A -> B.
"""
import argparse, csv, json, os, pickle, random, sys
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import COLS
from method2_sizes.genmodel import Simulator, is_short
from method2_sizes.decompose import parts
from method2_sizes.ceiling import predict, SHIFT


def views(full, mask, cut):
    obs = [t for t, m in enumerate(mask) if m]
    block = set(range(cut - 40, cut)); older = [t for t in obs if t not in block]
    rev = set(obs)
    for t in older:
        rev.update(range(max(0, t - 8), t))
    return ([v if m else -1 for v, m in zip(full, mask)], [v if t in rev else -1 for t, v in enumerate(full)], list(full), len(older))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkl"); ap.add_argument("params"); ap.add_argument("--data", default="data/main")
    ap.add_argument("--reps", type=int, default=4); a = ap.parse_args(); sys.setrecursionlimit(10000)
    pub = public_dir(a.data); m = pickle.load(open(a.pkl, "rb")); par = json.load(open(a.params))
    vq, vs = load_split(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    ref = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("reference.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    shorts = [r for r in vq if cond[r["query_id"]] == "short_history"]
    refmse = sum(sum((x - y) ** 2 for x, y in zip(ref[r["query_id"]], T[r["query_id"]])) for r in shorts) / len(shorts)
    print(f"{len(shorts)} real short-history templates x {a.reps} reps; reference-model mse on real short = {refmse:.4f}")
    sim = Simulator(par, state_win_shift=SHIFT, seed=33); rng = random.Random(3)
    acc = {v: dict(win=0.0, size=0.0, tot=0.0) for v in "ABC"}; n = 0
    d_pw, d_a, ndrop = [], [], []
    for rep in range(a.reps):
        for r in shorts:
            mask = [v >= 0 for v in vs[r["machine_id"]]]; cut = r["cutoff"]
            tau = round(rng.uniform(.3, .9) * cut) if rng.random() < .05 else None
            full, truth = sim.machine(r["machine_type"], r["signal"], cut, [True] * cut, tau=tau, with_truth=True)
            A, B, C, nold = views(full, mask, cut); ndrop.append(nold); preds = {}
            for name, s in zip("ABC", (A, B, C)):
                p = predict(m.win, m.sizes, r, s); preds[name] = p
                w, z, c, *_ = parts([x / sum(p) for x in p], truth)
                acc[name]["win"] += w; acc[name]["size"] += z + c; acc[name]["tot"] += sum((x / sum(p) - y) ** 2 for x, y in zip(p, truth))
            n += 1; d_pw.append(abs(preds["B"][0] - preds["A"][0])); d_a.append(abs(preds["B"][1] / (1 - preds["B"][0]) - preds["A"][1] / (1 - preds["A"][0])))
    print(f"\nmean sampled-older ticks per short machine = {sum(ndrop) / len(ndrop):.1f} (outcomes the gate currently skips)")
    print(f"\n{'view':26s} {'total mse':>10s} {'win part':>10s} {'size part':>10s}   as loss ratio vs reference")
    for name, label in (("A", "A current (real mask)"), ("B", "B lags revealed (upper bd)"), ("C", "C fully observed (oracle)")):
        t, w, z = (acc[name][k] / n for k in ("tot", "win", "size"))
        print(f"{label:26s} {t:10.5f} {w:10.5f} {z:10.5f}   {t / refmse:.4f}")
    gain = (acc["A"]["tot"] - acc["B"]["tot"]) / n
    print(f"\nrecoverable by marginalizing the sampled ticks (A - B) = {gain:.5f} mse = {gain / refmse:.4f} in short loss-ratio units"
          f" = {gain / refmse / 4:.4f} in R  ->  about {60 * (gain / refmse / 4) / (1.033) ** 2:+.3f} points")
    print(f"mask cost beyond that (B - C) = {(acc['B']['tot'] - acc['C']['tot']) / n:.5f}")
    for th in (.01, .02, .05):
        print(f"queries whose P(win) moves by > {th:.2f} between A and B: {sum(x > th for x in d_pw) / len(d_pw):.1%};  "
              f"P(1x|win) moves by > {th:.2f}: {sum(x > th for x in d_a) / len(d_a):.1%}")


if __name__ == "__main__":
    main()
