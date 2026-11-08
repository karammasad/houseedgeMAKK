"""Cross-machine recent-win adjustment (teammate's rule, reimplemented from its spec; beta fixed at 2.0, NOT re-tuned here).
f(m) = mean over the OTHER machines j of the episode of [win rate of j over ticks cutoff-8 .. cutoff-1 (observed only)]
       - [win rate of j over all observed ticks < cutoff]. P(win) of m moves on the logit scale by beta * f; size proportions kept.
Uses only pre-cutoff histories (every machine of an episode shares the cutoff)."""
import math
from collections import defaultdict


def adjust(queries, seq, pred, beta=2.0, L=8):
    by = defaultdict(list)
    for q in queries:
        by[q["episode_id"]].append(q)
    stat = {}
    for q in queries:
        y = seq[q["machine_id"]]; c = q["cutoff"]
        rec = [int(v > 0) for v in y[max(0, c - L):c] if v >= 0]; allw = [int(v > 0) for v in y[:c] if v >= 0]
        stat[q["machine_id"]] = (sum(rec) / len(rec) - sum(allw) / len(allw)) if rec and allw else None
    out = {}
    for ep, qs in by.items():
        for q in qs:
            d = [stat[o["machine_id"]] for o in qs if o is not q and stat[o["machine_id"]] is not None]
            p = list(pred[q["query_id"]])
            if not d:
                out[q["query_id"]] = p; continue
            f = sum(d) / len(d); pw = min(max(1 - p[0], 1e-9), 1 - 1e-9)
            pn = 1 / (1 + math.exp(-(math.log(pw / (1 - pw)) + beta * f)))
            out[q["query_id"]] = [1 - pn] + [x / pw * pn for x in p[1:]]
    return out
