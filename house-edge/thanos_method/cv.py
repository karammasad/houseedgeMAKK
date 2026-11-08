"""Train-only episode-grouped cross-validation (SURROGATE, not the official metric: train has no true probability vectors).

For each of 4 folds (episodes split by SHA-256 order within condition): fit EGP on the other 3 folds' TRAIN histories only, then
predict the held-out fold's machines at cutoffs c, c-1, c-2, c-3 (history truncated to ticks < cutoff; outcome = train label at c,
the observed history tick otherwise). Score: 5-class Brier (squared error to the one-hot outcome) and log loss, per condition and type.
Also reported: the same model fitted on ALL train (in-sample hyperparameters) on the same rows, and a pooled per-type frequency baseline
fitted on the training folds. Conditions / labels are read here for SCORING only (train split; permitted).
    python -m thanos_method.cv --data data/main --out <dir>
"""
import argparse, csv, hashlib, json, os, pickle, time
from collections import defaultdict
import numpy as np
from common.data import load_split, public_dir
from . import egp
from .model import Model

CONDS = ("stable", "short_history", "drift", "shared_change")
PAY = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}


def folds(queries, conds=None, k=4):
    eps = sorted({r["episode_id"] for r in queries}, key=lambda e: hashlib.sha256(e.encode()).hexdigest())
    if conds:
        byc = defaultdict(list)
        for e in eps:
            byc[conds[e]].append(e)
        F = [[] for _ in range(k)]
        for c in sorted(byc):
            for i, e in enumerate(byc[c]):
                F[i % k].append(e)
        return F
    return [eps[i::k] for i in range(k)]


def rows_for(machines, labels, lags=(0, 1, 2, 3)):
    """(machine truncated at c - j, outcome index) pairs."""
    out = []
    for m in machines:
        for j in lags:
            if j == 0:
                y = int(labels[m.qid]); mm = m
            else:
                c = m.cutoff - j; y = int(m.y[c])
                if y < 0:
                    continue
                mm = egp.Machine(m.qid, m.ep, m.mid, m.type, m.signal, m.y[:c])
            out.append((mm, PAY[y], j))
    return out


def score(P, y):
    oh = np.zeros_like(P); oh[np.arange(len(y)), y] = 1
    return ((P - oh) ** 2).sum(1), -np.log(np.maximum(P[np.arange(len(y)), y], 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--out", required=True); ap.add_argument("--tickset", default="own")
    a = ap.parse_args(); pub = public_dir(a.data); os.makedirs(a.out, exist_ok=True)
    q, seq = load_split(pub, "train"); ms = egp.machines_from_split(q, seq)
    with open(os.path.join(pub, "train", "labels.csv")) as f:
        labels = {r["query_id"]: r["outcome"] for r in csv.DictReader(f)}
    with open(os.path.join(pub, "train", "conditions.csv")) as f:
        cond_q = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}
    cond_e = {r["episode_id"]: cond_q[r["query_id"]] for r in q}
    F = folds(q, cond_e)
    res = []; t0 = time.time()
    for i, fe in enumerate(F):
        fe = set(fe); tr = [m for m in ms if m.ep not in fe]; te = [m for m in ms if m.ep in fe]
        assert not ({m.ep for m in tr} & {m.ep for m in te})
        M = Model("train", verbose=False, tickset=a.tickset).fit(None, machines=tr)
        R = rows_for(te, labels)
        P = np.array([egp.machine_posterior(mm, M.H)[1] for mm, _, _ in R])
        # pooled per-type frequency baseline from the training folds' observed payouts
        base = {}
        for t in {m.type for m in tr}:
            ys = np.concatenate([m.y[m.y >= 0] for m in tr if m.type == t]); base[t] = np.array([(ys == v).mean() for v in (0, 1, 2, 5, 20)])
        P0 = np.array([base[mm.type] for mm, _, _ in R])
        for (mm, y, j), p, p0 in zip(R, P, P0):
            res.append(dict(fold=i, qid=mm.qid, ep=mm.ep, type=mm.type, cond=cond_e[mm.ep], lag=j, y=y, p=p.tolist(), p0=p0.tolist()))
        print(f"fold {i}: fit on {len(tr)} machines, {len(R)} held-out rows, {time.time() - t0:.0f} s", flush=True)
    # in-sample reference: hyperparameters from all train
    Mall = Model("train", verbose=False, tickset=a.tickset).fit(None, machines=ms)
    byq = {m.qid: m for m in ms}
    Rall = rows_for(ms, labels); Pin = {(mm.qid, j): egp.machine_posterior(mm, Mall.H)[1] for mm, _, j in Rall}
    for r in res:
        r["pin"] = Pin[(r["qid"], r["lag"])].tolist()
    pickle.dump(res, open(os.path.join(a.out, "cv_rows.pkl"), "wb"))
    y = np.array([r["y"] for r in res])
    out = {}
    for name, key in (("EGP out-of-fold", "p"), ("EGP in-sample hyper", "pin"), ("pooled per-type freq", "p0")):
        P = np.array([r[key] for r in res]); b, l = score(P, y)
        line = {"brier": float(b.mean()), "logloss": float(l.mean())}
        for c in CONDS:
            s = np.array([r["cond"] == c for r in res]); line[c] = float(b[s].mean())
        for t in ("balanced", "jackpot", "steady"):
            s = np.array([r["type"] == t for r in res]); line[t] = float(b[s].mean())
        out[name] = line
        print(f"{name:24s} brier {line['brier']:.5f} logloss {line['logloss']:.5f} | " + " ".join(f"{c} {line[c]:.4f}" for c in CONDS) +
              " | " + " ".join(f"{t} {line[t]:.4f}" for t in ("balanced", "jackpot", "steady")))
    # paired episode bootstrap: out-of-fold vs in-sample hyperparameters (generalization gap of the hyperparameter fit)
    bo, _ = score(np.array([r["p"] for r in res]), y); bi, _ = score(np.array([r["pin"] for r in res]), y)
    eps = sorted({r["ep"] for r in res}); idx = defaultdict(list)
    for i, r in enumerate(res):
        idx[r["ep"]].append(i)
    d = np.array([np.mean(bo[idx[e]] - bi[idx[e]]) for e in eps]); rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    out["oof_minus_insample"] = dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    print(f"out-of-fold minus in-sample Brier per episode: {d.mean():+.6f}  95% CI [{np.percentile(bs, 2.5):+.6f}, {np.percentile(bs, 97.5):+.6f}]")
    json.dump(out, open(os.path.join(a.out, "cv_summary.json"), "w"), indent=1)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
