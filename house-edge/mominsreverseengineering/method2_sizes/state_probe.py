"""Probe: what predicts the binary size state (P(1x|win) high vs low) at the cutoff?
Diagnosis only: uses validation TUNING-half targets to label the state; run from repo root:
  python3 -m method2_sizes.state_probe --data data/main
"""
import argparse, csv, itertools, os
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import folds


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def wins_1x(s, upto=None):
    """List of (tick, is_1x) for observed wins before `upto`."""
    upto = len(s) if upto is None else upto
    return [(t, int(s[t] == 1)) for t in range(upto) if s[t] > 0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub)
    q, seq = load_split(pub, "validation")
    with open(os.path.join(pub, "validation", "targets.csv")) as f:
        T = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in csv.DictReader(f)}
    rows = []
    for r in q:
        if r["episode_id"] not in tune:
            continue
        p = T[r["query_id"]]; w = 1 - p[0]
        if w < 1e-6:
            continue
        rows.append((r, p[1] / w))
    feats = defaultdict(list); lab = []
    for r, a1 in rows:
        s = seq[r["machine_id"]]; cut = len(s)
        others = [x for rr, x in rows if rr["machine_type"] == r["machine_type"]]
        lab.append(a1 > (min(others) + max(others)) / 2)   # high-1x state
        W = wins_1x(s)
        for n in (3, 5, 10, 20, 40, 10 ** 6):
            last = W[-n:]
            feats[f"1x share, last {n if n < 10**6 else 'all'} wins"].append(sum(x for _, x in last) / max(1, len(last)))
        for span in (10, 20, 40):
            last = [x for t, x in W if t >= cut - span]
            feats[f"1x share, wins in last {span} ticks"].append(sum(last) / len(last) if last else .5)
        # per-machine signed lag rules: P(1x | lag-k win == current) estimated from this machine's past
        for k in range(1, 9):
            cur = s[cut - k]
            if cur < 0:
                feats[f"machine-fitted lag{k} win/loss"].append(.5); continue
            c = [x for t, x in W if t - k >= 0 and s[t - k] >= 0 and (s[t - k] > 0) == (cur > 0)]
            feats[f"machine-fitted lag{k} win/loss"].append((sum(c) + 1) / (len(c) + 2))
        for a_, b_ in itertools.combinations(range(1, 9), 2):
            ca, cb = s[cut - a_], s[cut - b_]
            if ca < 0 or cb < 0:
                feats[f"machine-fitted xnor{a_}{b_}"].append(.5); continue
            x0 = (ca > 0) == (cb > 0)
            c = [x for t, x in W if t - b_ >= 0 and s[t - a_] >= 0 and s[t - b_] >= 0 and ((s[t - a_] > 0) == (s[t - b_] > 0)) == x0]
            feats[f"machine-fitted xnor{a_}{b_}"].append((sum(c) + 1) / (len(c) + 2))
    res = sorted(((auc(v, lab), k) for k, v in feats.items()), reverse=True)
    print(f"n={len(lab)} queries, high-1x state rate {sum(lab) / len(lab):.2f}")
    print("AUC for predicting the high-1x size state (0.5 = useless, 1 = perfect):")
    for v, k in res[:14]:
        print(f"  {v:.3f}  {k}")
    print("  ...")
    for v, k in res[-3:]:
        print(f"  {v:.3f}  {k}")


if __name__ == "__main__":
    main()
