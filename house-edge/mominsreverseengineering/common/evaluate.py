"""Shared evaluator. Every method is scored the same way, on the kit's tuning/audit split.

Usage (from the repo root):
  python -m common.evaluate --method method1_gate --data data/main --note "baseline"
  python -m common.evaluate --method method1_gate --data data/main --audit          # also prints the audit half
  python -m common.evaluate --method method1_gate --data data/main --test-out out/submission.csv

A method is a package with model.py defining class Model:
  Model(**kwargs).fit(pub)              learns from pub/train (and may read other splits' histories, never labels/targets)
  Model.predict(pub, split) -> dict     {query_id: [p0, p1, p2, p5, p20]}
Results are appended to experiments/results.csv.
"""
import argparse, csv, hashlib, importlib, json, math, os, time
from collections import defaultdict
from .data import public_dir, write_predictions

CONDITIONS = ("stable", "short_history", "drift", "shared_change")
COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def folds(pub):
    """The kit's split (slot_forecasting/experiments.py): per condition, episodes sorted by SHA-256, alternating."""
    eps = defaultdict(set)
    qc = {r["query_id"]: r["condition"] for r in _rows(os.path.join(pub, "validation", "conditions.csv"))}
    for q in _rows(os.path.join(pub, "validation", "queries.csv")):
        eps[qc[q["query_id"]]].add(q["episode_id"])
    tune, audit = set(), set()
    for c in CONDITIONS:
        order = sorted(eps[c], key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        tune.update(order[::2]); audit.update(order[1::2])
    return tune, audit


def points(pub, pred, episodes=None, max_points=60.0):
    """Official contest metric: squared error per query, mean per episode, mean per condition,
    divided by the frozen reference's error per condition; R = mean ratio; points = max/(1+R)."""
    targets = _rows(os.path.join(pub, "validation", "targets.csv"))
    cond = {r["query_id"]: r["condition"] for r in _rows(os.path.join(pub, "validation", "conditions.csv"))}
    ref = {r["query_id"]: [float(r[c]) for c in COLS] for r in _rows(os.path.join(pub, "validation", "reference.csv"))}
    err = defaultdict(lambda: defaultdict(list)); rerr = defaultdict(lambda: defaultdict(list))
    for t in targets:
        if episodes is not None and t["episode_id"] not in episodes:
            continue
        truth = [float(t[c]) for c in COLS]; p = pred[t["query_id"]]; s = sum(p); p = [x / s for x in p]
        c = cond[t["query_id"]]
        err[c][t["episode_id"]].append(sum((a - b) ** 2 for a, b in zip(p, truth)))
        rerr[c][t["episode_id"]].append(sum((a - b) ** 2 for a, b in zip(ref[t["query_id"]], truth)))
    ratios = {}
    for c in CONDITIONS:
        e = sum(sum(v) / len(v) for v in err[c].values()) / len(err[c])
        b = sum(sum(v) / len(v) for v in rerr[c].values()) / len(rerr[c])
        ratios[c] = e / max(b, 1e-9)
    R = sum(ratios.values()) / len(ratios)
    return max_points / (1 + R), ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="package name, e.g. method1_gate")
    ap.add_argument("--data", default="data/main")
    ap.add_argument("--kwargs", default="{}", help='JSON settings passed to Model(), e.g. \'{"size_strength": 30}\'')
    ap.add_argument("--note", default="")
    ap.add_argument("--audit", action="store_true", help="also score the audit half (do this once per final candidate)")
    ap.add_argument("--test-out", help="also predict the test split and write this CSV")
    a = ap.parse_args()
    pub = public_dir(a.data)
    with open(os.path.join(pub, "metadata.json")) as f:
        maxp = float(json.load(f).get("max_points", 60))
    t0 = time.time()
    model = importlib.import_module(f"{a.method}.model").Model(**json.loads(a.kwargs)).fit(pub)
    pred = model.predict(pub, "validation")
    secs = time.time() - t0
    tune, audit = folds(pub)
    tp, tr = points(pub, pred, tune, maxp)
    ap_ = points(pub, pred, audit, maxp)[0] if a.audit else None
    print(f"{a.method} {a.kwargs}: tuning {tp:.3f}" + (f"  audit {ap_:.3f}" if a.audit else "") + f"  ({secs:.1f}s)")
    print("  tuning loss ratios: " + ", ".join(f"{c} {tr[c]:.4f}" for c in CONDITIONS))
    os.makedirs("experiments", exist_ok=True)
    path = os.path.join("experiments", "results.csv"); new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "user", "method", "kwargs", "note", "tuning", "audit"] + [f"tune_{c}" for c in CONDITIONS] + ["seconds"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), os.environ.get("USER", ""), a.method, a.kwargs, a.note,
                    f"{tp:.4f}", "" if ap_ is None else f"{ap_:.4f}"] + [f"{tr[c]:.5f}" for c in CONDITIONS] + [f"{secs:.1f}"])
    if a.test_out:
        write_predictions(a.test_out, model.predict(pub, "test"))
        print("  wrote", a.test_out, "- check it with: python kit/check_submission.py", a.test_out, "--data", pub)


if __name__ == "__main__":
    main()
