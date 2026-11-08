"""Shared data loading (standard library only).

A split is returned as (queries, seq):
  queries: list of dicts with query_id, episode_id, machine_id, machine_type, signal (int), cutoff (int)
  seq:     {machine_id: list of length cutoff}, seq[m][t] = payout at tick t, or -1 if the tick is missing.
Every value in seq is from a tick strictly before that machine's cutoff, so nothing here can leak.
"""
import csv, os

PAYOUTS = (0, 1, 2, 5, 20)
SIZES = (1, 2, 5, 20)


def public_dir(path):
    """Accept the kit folder (containing public/) or the public/ folder itself."""
    p = os.path.join(path, "public")
    return p if os.path.isdir(p) else path


def load_split(pub, split):
    with open(os.path.join(pub, split, "queries.csv"), newline="") as f:
        queries = list(csv.DictReader(f))
    for r in queries:
        r["cutoff"] = int(r["cutoff"])
        r["signal"] = int(r["signal"])
    seq = {r["machine_id"]: [-1] * r["cutoff"] for r in queries}
    with open(os.path.join(pub, split, "history.csv"), newline="") as f:
        for r in csv.DictReader(f):
            seq[r["machine_id"]][int(r["tick"])] = int(r["payout"])
    return queries, seq


def wins(seq_m):
    """Win indicator per tick: 1 win, 0 loss, -1 missing."""
    return [-1 if v < 0 else int(v > 0) for v in seq_m]


def write_predictions(path, pred):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for k in sorted(pred):
            w.writerow([k] + [format(v, ".17g") for v in pred[k]])
