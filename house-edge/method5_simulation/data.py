"""Method 5 data layer: causal windows and statics from a machine's own history (numpy; research/training side).

Every feature of an example whose target is tick t is a function of cats[:t] ONLY (cats = payout category per tick,
-1 = unobserved). The builder never receives the cutoff, later ticks, other machines, labels of other queries or any
prediction of Methods 1-4. Inputs: observed categories, observation flags, absolute tick, window position, machine
type and signal, and prefix / recent empirical payout frequencies.

Category index: payout 0,1,2,5,20 -> 0..4.   Static vector (17): type one-hot (3), signal, t/256, prefix category
frequencies (5), log1p(n_obs)/6, recent-32 category frequencies (5), n_recent/32.
"""
import numpy as np
from common.data import load_split, public_dir

PAY = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
TYPES = ("steady", "balanced", "jackpot")
L = 64            # window of the previous L ticks (dense grid; unobserved ticks are flagged, never zeros)
NS = 17


def machine_cats(seq_m):
    return np.array([-1 if v < 0 else PAY[v] for v in seq_m], dtype=np.int8)


def is_short(cats):
    return (cats >= 0).mean() < 0.6


def target_ticks(cats):
    """Self-supervised targets: observed ticks in the final 40% of a long history; the final 40-tick block of a short
    one (so training mirrors the query: late in the history, post any regime change)."""
    cut = len(cats)
    lo = cut - 40 if is_short(cats) else int(0.6 * cut)
    return [t for t in range(max(lo, 8), cut) if cats[t] >= 0]


def build(cats, ts, typ, signal):
    """Windows (N, L) of categories for ticks t-L..t-1 (-1 = unobserved / before tick 0), statics (N, NS)."""
    ts = np.asarray(ts); n = len(ts)
    cum = np.zeros((len(cats) + 1, 5), dtype=np.float64)
    for k in range(5):
        cum[1:, k] = np.cumsum(cats == k)
    idx = ts[:, None] + np.arange(-L, 0)[None, :]
    W = np.where(idx >= 0, cats[np.clip(idx, 0, None)], -1).astype(np.int8)
    S = np.zeros((n, NS), dtype=np.float32)
    S[:, TYPES.index(typ)] = 1.0; S[:, 3] = signal; S[:, 4] = ts / 256.0
    pre = cum[ts]; npre = pre.sum(1)
    S[:, 5:10] = pre / np.maximum(npre, 1)[:, None]; S[:, 10] = np.log1p(npre) / 6.0
    rec = cum[ts] - cum[np.maximum(ts - 32, 0)]; nrec = rec.sum(1)
    S[:, 11:16] = rec / np.maximum(nrec, 1)[:, None]; S[:, 16] = nrec / 32.0
    return W, S


def load_machines(pub, split):
    """List of dicts per machine: query row, cats; train queries also carry the observed label category."""
    q, seq = load_split(pub, split); out = []
    for r in q:
        out.append(dict(qid=r["query_id"], ep=r["episode_id"], mid=r["machine_id"], typ=r["machine_type"],
                        sig=int(r["signal"]), cut=r["cutoff"], cats=machine_cats(seq[r["machine_id"]])))
    return out


def train_labels(pub):
    import csv, os
    return {r["query_id"]: PAY[int(r["outcome"])] for r in csv.DictReader(open(os.path.join(pub, "train", "labels.csv")))}


def examples(machines, labels=None, with_queries=True):
    """Stack all self-supervised examples (+ the labelled cutoff queries if labels given). Returns dict of arrays with
    per-example machine index, episode, short flag, and `is_query`."""
    Ws, Ss, Ys, mi, isq = [], [], [], [], []
    for i, m in enumerate(machines):
        ts = target_ticks(m["cats"])
        if ts:
            W, S = build(m["cats"], ts, m["typ"], m["sig"]); Ws.append(W); Ss.append(S)
            Ys.append(m["cats"][ts].astype(np.int64)); mi += [i] * len(ts); isq += [0] * len(ts)
        if with_queries and labels is not None:
            W, S = build(m["cats"], [m["cut"]], m["typ"], m["sig"]); Ws.append(W); Ss.append(S)
            Ys.append(np.array([labels[m["qid"]]])); mi.append(i); isq.append(1)
    mi = np.array(mi)
    return dict(W=np.concatenate(Ws), S=np.concatenate(Ss), y=np.concatenate(Ys), mi=mi, is_query=np.array(isq),
                ep=np.array([machines[i]["ep"] for i in mi]), short=np.array([is_short(machines[i]["cats"]) for i in mi]))
