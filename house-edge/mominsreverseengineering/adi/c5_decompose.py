"""C5 diagnostic: split each condition's error into win-probability vs size parts (analysis only; reads targets).

Swaps in the TRUE win probability (keeping our size split), or the TRUE size split (keeping our win probability),
and reports the resulting loss ratios. Usage: python -m adi.c5_decompose --data data/main --kwargs '{...}'
"""
import argparse, json, csv, os
from common.data import public_dir
from common.evaluate import points, folds, COLS
from adi.model import Model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--kwargs", default="{}")
    a = ap.parse_args(); pub = public_dir(a.data)
    pred = Model(**json.loads(a.kwargs)).fit(pub).predict(pub, "validation")
    with open(os.path.join(pub, "validation", "targets.csv")) as f:
        truth = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(f)}
    tune, _ = folds(pub)
    def swap(kind):
        out = {}
        for k, p in pred.items():
            t = truth[k]; pw, tw = 1 - p[0], 1 - t[0]
            sz = [x / pw for x in p[1:]] if pw > 0 else [.25] * 4
            tsz = [x / tw for x in t[1:]] if tw > 0 else [.25] * 4
            if kind == "true_win":
                out[k] = [t[0]] + [tw * s for s in sz]
            else:
                out[k] = [p[0]] + [pw * s for s in tsz]
        return out
    for name, pr in (("ours", pred), ("true win prob, our sizes", swap("true_win")), ("our win prob, true sizes", swap("true_size"))):
        pts, r = points(pub, pr, tune)
        print(f"{name:26s} {pts:6.2f}  " + "  ".join(f"{c} {v:.4f}" for c, v in r.items()))


if __name__ == "__main__":
    main()
