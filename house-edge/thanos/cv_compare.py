"""Paired comparison of two CV row files from thanos.cv (same folds, same rows), per condition.
Decision rule (LOG.md section 4): the new tick set replaces the old one only if short_history out-of-fold Brier is lower with paired
episode-bootstrap P >= .9 and no other condition is worse.
    python -m thanos.cv_compare --new <cv_own>/cv_rows.pkl --old <cv_common>/cv_rows.pkl
"""
import argparse, pickle
from collections import defaultdict
import numpy as np

CONDS = ("stable", "short_history", "drift", "shared_change")


def brier(p, y):
    oh = np.zeros(5); oh[y] = 1; return float(((np.asarray(p) - oh) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--old", required=True); a = ap.parse_args()
    new, old = pickle.load(open(a.new, "rb")), pickle.load(open(a.old, "rb"))
    key = lambda r: (r["qid"], r["lag"])
    O = {key(r): r for r in old}; assert len(O) == len(new) and all(key(r) in O for r in new), "row sets differ"
    assert all(O[key(r)]["fold"] == r["fold"] and O[key(r)]["y"] == r["y"] for r in new), "folds / outcomes differ"
    rng = np.random.default_rng(0); verdict = {}
    for c in CONDS + ("all",):
        by = defaultdict(list)
        for r in new:
            if c == "all" or r["cond"] == c:
                by[r["ep"]].append(brier(r["p"], r["y"]) - brier(O[key(r)]["p"], r["y"]))
        d = np.array([np.mean(v) for v in by.values()]); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
        verdict[c] = (d.mean(), (bs < 0).mean())
        print(f"{c:14s} episodes {len(d):3d}  new - old OOF Brier {d.mean():+.6f}  95% CI [{np.percentile(bs, 2.5):+.6f}, {np.percentile(bs, 97.5):+.6f}]"
              f"  P(new better) {(bs < 0).mean():.3f}")
    ok = verdict["short_history"][0] < 0 and verdict["short_history"][1] >= .9 and all(verdict[c][0] <= 1e-12 for c in CONDS if c != "short_history")
    print("VERDICT (section-4 rule):", "PASS - new tick set replaces the old" if ok else "FAIL - keep the old tick set")


if __name__ == "__main__":
    main()
