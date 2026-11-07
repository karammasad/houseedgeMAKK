"""Experiment 10: uncertainty-adaptive size -> win feedback vs the short-log rule (tuning half; no audit).

  python -m method2_sizes.adaptive_eval --data data/main

Model = current FINAL (adi size_split + cell_signal + method 2 joint-EM sizes). Learned gate fitted on TRAIN only.
Diagnostics: where full feedback helps, by win-posterior entropy tercile x short/long, and whether disagreement
(win-only vs joint posterior) predicts the gain.
"""
import argparse, csv, os, time
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import folds, points, COLS, CONDITIONS
from method2_sizes.round2 import FINAL
from method2_sizes.model import Model
from method2_sizes.joint import joint_parts, assemble, lag_prior_rows
from method2_sizes.decompose import parts
from method2_sizes import adaptive as A


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub)
    t0 = time.time(); m = Model(**FINAL).fit(pub); print(f"FINAL fit {time.time() - t0:.0f}s")
    tq, ts = load_split(pub, "train")
    with open(os.path.join(pub, "train", "labels.csv")) as f:
        lab = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(f)}
    t0 = time.time(); ex = A.training_examples(m, tq, ts, labels=lab)
    th, ll = A.fit_learned_gate(ex)
    print(f"learned gate on {len(ex)} train examples ({time.time() - t0:.0f}s): lam = sigmoid({th[0]:+.3f} "
          f"{th[1]:+.3f}*H {th[2]:+.3f}*short)  ->  long H=0: {A.sigmoid(th[0]):.2f}, long H=1: "
          f"{A.sigmoid(th[0] + th[1]):.2f}, short H=0: {A.sigmoid(th[0] + th[2]):.2f}, short H=1: "
          f"{A.sigmoid(sum(th)):.2f}")
    vq, vs = load_split(pub, "validation"); rows = lag_prior_rows(m.win, m.sizes)
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    gates = ["none", "short", "entropy", "learned", "full"]; V = {g: {} for g in gates}; diag = []
    for r in vq:
        s = vs[r["machine_id"]]; P = joint_parts(m.win, m.sizes, rows, r, s); H, sh = A.features(P, s)
        lam = {"none": 0.0, "full": 1.0, "short": sh, "entropy": H, "learned": A.lam_of("learned", th, H, sh)}
        for g in gates:
            p, post = assemble(m.sizes, rows, r, s, P, lam[g] > 0, False, lam[g]); V[g][r["query_id"]] = p
            if g == "full":
                dis = .5 * sum(abs(x - y) for x, y in zip(post, P["post_s"]))
        if r["episode_id"] in tune:
            e0 = parts(V["none"][r["query_id"]], T[r["query_id"]])[0]; e1 = parts(V["full"][r["query_id"]], T[r["query_id"]])[0]
            diag.append((H, sh, dis, e0 - e1, cond[r["query_id"]]))
    print(f"\n{'gate':10s} {'tuning':>7s} {'vs none':>8s}  " + "  ".join(f"{c[:6]:>6s}" for c in CONDITIONS))
    base = points(pub, V["none"], tune)[0]
    for g in gates:
        tp, tr = points(pub, V[g], tune)
        print(f"{g:10s} {tp:7.3f} {tp - base:+8.3f}  " + "  ".join(f"{tr[c]:.4f}" for c in CONDITIONS))
    print("\nwin-part error reduction from FULL feedback (positive = feedback helps), tuning queries:")
    hs = sorted(x[0] for x in diag); cut1, cut2 = hs[len(hs) // 3], hs[2 * len(hs) // 3]
    by = defaultdict(list)
    for H, sh, dis, g, c in diag:
        by[("short" if sh else "long", 0 if H < cut1 else 1 if H < cut2 else 2)].append(g)
    for k in sorted(by):
        v = by[k]; print(f"  {k[0]:5s} entropy tercile {k[1]} (n={len(v):3d}): mean gain {sum(v) / len(v):+.5f}, "
                         f"helps on {sum(x > 0 for x in v)}/{len(v)}")
    ds = sorted(diag, key=lambda x: x[2]); n = len(ds)
    for lo, hi in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)):
        v = [x[3] for x in ds[lo:hi]]
        print(f"  disagreement {ds[lo][2]:.2f}-{ds[hi - 1][2]:.2f}: mean gain {sum(v) / len(v):+.5f}")
    print(f"\nentropy terciles cut at H = {cut1:.3f}, {cut2:.3f}")


if __name__ == "__main__":
    main()
