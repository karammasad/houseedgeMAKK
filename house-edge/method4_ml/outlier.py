"""DPCM: discounted prequential context mixing (Round 2 independent challenger; not a gate / rates / flip model).

Universal-prediction view: each machine's payout stream is forecast by a mixture of simple 'experts'. Expert e = a context made
of 0, 1 or 2 binary features of the last 8 ticks (feature = own win at lag l, or own payout >= 2 at lag l; 16 features ->
1 + 16 + 120 = 137 experts). Expert e predicts the 5-class next payout from DISCOUNTED Dirichlet counts in its current context
cell, centred on a population prior (pooled 5-class frequency of the machine's (type, signal) group from the round's TRAIN
histories, strength alpha). Experts are weighted by exp(discounted prequential log-likelihood): every observed tick is
predicted by each expert from strictly earlier ticks only, then counted. Forgetting (lambda) replaces an explicit flip model.
Missing ticks are skipped (never treated as losses); an expert whose context is unobservable predicts its cell-marginal.
Free parameters: lambda (count + loss forgetting) and alpha (prior strength). Nothing else is fitted.

  python3 -m method4_ml.outlier --data data/adaptation --out build/dpcm
"""
import argparse, itertools as it, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load_split, public_dir, write_predictions  # noqa: E402

CAT = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
L = 8
FEATS = [("w", l) for l in range(1, L + 1)] + [("b", l) for l in range(1, L + 1)]
EXPERTS = [()] + [(i,) for i in range(16)] + list(it.combinations(range(16), 2))


def arrays(queries, seq):
    N = len(queries); T = max(max(q["cutoff"], len(seq[q["machine_id"]])) for q in queries) + 1
    Y = np.full((N, T), -1, np.int8)
    for i, q in enumerate(queries):
        y = seq[q["machine_id"]]
        Y[i, :len(y)] = [CAT[v] if v >= 0 else -1 for v in y]
    cut = np.array([q["cutoff"] for q in queries])
    return Y, cut


def feature_bits(Y):
    """B[f, n, t] in {0, 1, -1}: feature f evaluated for predicting tick t, from tick t - lag only (strictly earlier)."""
    N, T = Y.shape; B = np.full((16, N, T), -1, np.int8)
    for f, (k, l) in enumerate(FEATS):
        src = Y[:, :T - l]; v = np.where(src < 0, -1, (src >= 1) if k == "w" else (src >= 2)).astype(np.int8)
        B[f, :, l:] = v
    return B


def prior_table(qs_seqs):
    """Pooled 5-class frequency per (type, signal) over TRAIN histories (+1 smoothing); fallback per type."""
    tab = {}
    for queries, seq in qs_seqs:
        for q in queries:
            y = [CAT[v] for v in seq[q["machine_id"]] if v >= 0]
            for key in ((q["machine_type"], q["signal"]), (q["machine_type"], None)):
                tab.setdefault(key, np.ones(5)); tab[key] += np.bincount(y, minlength=5)
    return {k: v / v.sum() for k, v in tab.items()}


LAMS = (1.0, 0.98, 0.95)   # memory lengths mixed as experts (improvement 1: Bayesian mixture over forgetting rates)


def predict(queries, seq, prior, lam=None, alpha=4.0, mu=0.99, lams=LAMS):
    """lam given -> single forgetting rate (v0). Otherwise experts = contexts x lams, mixed with ONE common loss discount mu."""
    if lam is not None:
        pf, Lk = core(queries, seq, prior, lam, alpha, lam)
    else:
        parts = [core(queries, seq, prior, l, alpha, mu) for l in lams]
        pf = np.concatenate([a for a, _ in parts], 1); Lk = np.concatenate([b for _, b in parts], 1)
    w = np.exp(Lk - Lk.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    out = (w[:, :, None] * pf).sum(1); out = np.maximum(out, 1e-12); out /= out.sum(1, keepdims=True)
    return {q["query_id"]: out[i].tolist() for i, q in enumerate(queries)}


def core(queries, seq, prior, lam, alpha, mu):
    """Per expert: causal prediction at the cutoff (N, E, 5) and mu-discounted prequential log-likelihood (N, E)."""
    Y, cut = arrays(queries, seq); N, T = Y.shape; B = feature_bits(Y); E = len(EXPERTS)
    P0 = np.stack([prior.get((q["machine_type"], q["signal"]), prior[(q["machine_type"], None)]) for q in queries]).astype(np.float64)
    cell = np.zeros((E, N, T), np.int8); ok = np.ones((E, N, T), bool)
    for e, fs in enumerate(EXPERTS):
        for j, f in enumerate(fs):
            cell[e] += (np.maximum(B[f], 0) << j).astype(np.int8); ok[e] &= B[f] >= 0
    C = np.zeros((N, E, 4, 5)); Lk = np.zeros((N, E)); Lf = np.zeros((N, E)); out = np.zeros((N, E, 5)); ar = np.arange(N)
    aP = alpha * P0[:, None, :]
    for t in range(T):
        ce = cell[:, :, t].T.astype(np.int64); vk = ok[:, :, t].T              # (N, E)
        Cc = np.take_along_axis(C, ce[:, :, None, None], 2)[:, :, 0]           # counts in the current cell (N, E, 5)
        Cm = C.sum(2)                                                          # cell-marginal counts
        Cu = np.where(vk[:, :, None], Cc, Cm)
        p = (Cu + aP) / (Cu.sum(-1, keepdims=True) + alpha)                    # (N, E, 5) causal prediction for tick t
        fin = cut == t                                                         # query tick: keep every expert's forecast
        if fin.any():
            out[fin] = p[fin]; Lf[fin] = Lk[fin]                                 # snapshot at the cutoff
        y = Y[:, t].astype(np.int64); obs = (y >= 0) & (t < cut)
        Lk *= mu; C *= lam                                                    # forgetting per elapsed tick (missing ticks too)
        if obs.any():
            n = ar[obs]; yy = y[obs]
            Lk[n] += np.log(np.take_along_axis(p[n], yy[:, None, None].repeat(E, 1), 2)[:, :, 0])   # scored BEFORE counting y_t
            upd = vk[n]                                                        # count only where the context is observable
            nn, ee = np.nonzero(upd)
            C[n[nn], ee, ce[n[nn], ee], yy[nn]] += 1.0
    return out, Lf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/data"); ap.add_argument("--out", default="/output/submission.csv")
    ap.add_argument("--lam", type=float, default=None); ap.add_argument("--alpha", type=float, default=4.0); ap.add_argument("--mu", type=float, default=0.99)
    ap.add_argument("--also", default=""); a = ap.parse_args(); t0 = time.time()
    pub = public_dir(a.data); prior = prior_table([load_split(pub, "train")])
    for sp in ["test"] + [s for s in a.also.split(",") if s]:
        P = predict(*load_split(pub, sp), prior, a.lam, a.alpha, a.mu)
        path = a.out if sp == "test" else os.path.join(os.path.dirname(os.path.abspath(a.out)), f"{sp}.csv")
        write_predictions(path, P); print("wrote", path, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
