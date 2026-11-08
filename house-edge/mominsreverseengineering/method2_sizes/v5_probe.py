"""Where is Kareem's method 4 v5 still wrong on P(win)? Residual diagnostics (validation truth for ANALYSIS only).

  python -m method2_sizes.v5_probe --data data/main

Fits method4_ml v5 (read-only import, default settings), then on the validation TUNING half relates the residual
r = true P(win) - v5 P(win) to observables built from ticks strictly before the cutoff:
  pw          v5's own P(win) (calibration)
  old_rate    win rate on the sampled OLDER ticks whose lags are unobserved (short logs; currently unused by the gate)
  cell_recent recent (last 30 ticks) minus full-history win rate inside the machine's current MAP gate cell
  a1x         Method 2's P(1x | win) (size state), type-centred
  signal      machine signal
  ep_surprise mean over the OTHER machines in the episode of (recent observed win rate - their v5 P(win))
Reports, per condition and for short/long, the share of residual variance a binned version of each feature explains.
"""
import argparse, csv, math, os
from collections import defaultdict
from common.data import load_split, public_dir, wins
from common.evaluate import folds, COLS, CONDITIONS
import importlib


def binned_r2(x, r, nb=5):
    """Variance of r explained by bin means of x (quantile bins); 0 = useless, 1 = explains all."""
    if len(x) < 2 * nb:
        return float("nan")
    order = sorted(range(len(x)), key=lambda i: x[i]); mu = sum(r) / len(r)
    tot = sum((v - mu) ** 2 for v in r) or 1e-12; ex = 0.0
    for b in range(nb):
        idx = order[b * len(x) // nb:(b + 1) * len(x) // nb]
        m = sum(r[i] for i in idx) / len(idx); ex += len(idx) * (m - mu) ** 2
    return ex / tot


def corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy + 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub)
    M4 = importlib.import_module("method4_ml.model").Model().fit(pub)
    pred = M4.predict(pub, "validation")
    vq, vs = load_split(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    from adi.model import is_short
    from method2_sizes.round2 import FINAL
    from method2_sizes.model import Model as M2
    sizes = M2(**dict(FINAL, win_model=None)).sizes; tq, ts = load_split(pub, "train"); sizes.fit(tq, ts)
    tmean = defaultdict(list)
    pwv = {k: 1 - p[0] / sum(p) for k, p in pred.items()}
    by_ep = defaultdict(list)
    for r in vq:
        by_ep[r["episode_id"]].append(r)
    rows = []
    for r in vq:
        s = vs[r["machine_id"]]; w = wins(s); k = r["query_id"]; cut = len(s)
        recent = [x for x in w[-30:] if x >= 0]
        # older ticks whose lag-8 window is not fully observed (outside the consecutive block)
        old = [w[t] for t in range(cut) if w[t] >= 0 and any(w[t - l] < 0 for l in range(1, 9) if t - l >= 0) and t < cut - 40]
        a1 = sizes.predict(r, s)[0]; tmean[r["machine_type"]].append(a1)
        others = [o for o in by_ep[r["episode_id"]] if o["query_id"] != k]
        sur = [(sum(x for x in wins(vs[o["machine_id"]])[-30:] if x >= 0) /
                max(1, sum(1 for x in wins(vs[o["machine_id"]])[-30:] if x >= 0))) - pwv[o["query_id"]] for o in others]
        rows.append(dict(k=k, c=cond[k], short=is_short(s), r=(1 - T[k][0]) - pwv[k], pw=pwv[k],
                         old_rate=(sum(old) / len(old)) if old else float("nan"), n_old=len(old),
                         cell_recent=(sum(recent) / len(recent) if recent else .5) - (sum(x for x in w if x >= 0) / max(1, sum(1 for x in w if x >= 0))),
                         a1x=a1, typ=r["machine_type"], signal=r["signal"], ep_surprise=sum(sur) / len(sur) if sur else 0.0,
                         tune=r["episode_id"] in tune))
    for x in rows:
        x["a1x"] -= sum(tmean[x["typ"]]) / len(tmean[x["typ"]])
    R = [x for x in rows if x["tune"]]
    print(f"v5 residual r = true P(win) - v5 P(win) on {len(R)} tuning queries; mean squared r by group:")
    for c in CONDITIONS:
        v = [x["r"] ** 2 for x in R if x["c"] == c]; print(f"  {c:14s} mean r^2 {sum(v) / len(v):.5f}  (n={len(v)})")
    for sh in (True, False):
        v = [x["r"] ** 2 for x in R if x["short"] == sh]; print(f"  {'short' if sh else 'long':14s} mean r^2 {sum(v) / len(v):.5f}  (n={len(v)})")
    print("\nshare of residual variance explained by a 5-bin version of each feature (and linear corr):")
    feats = ["pw", "old_rate", "cell_recent", "a1x", "signal", "ep_surprise"]
    for grp, sel in (("all", lambda x: True), ("short", lambda x: x["short"]), ("long", lambda x: not x["short"])):
        G = [x for x in R if sel(x)]
        line = []
        for f in feats:
            H = [x for x in G if not (isinstance(x[f], float) and math.isnan(x[f]))]
            if len(H) < 20:
                line.append(f"{f} n/a"); continue
            line.append(f"{f} {binned_r2([x[f] for x in H], [x['r'] for x in H]):.3f} ({corr([x[f] for x in H], [x['r'] for x in H]):+.2f})")
        print(f"  {grp:5s}: " + "  ".join(line))
    for t in ("jackpot", "balanced", "steady"):
        G = [x for x in R if x["typ"] == t]
        print(f"  {t:8s} mean r^2 {sum(x['r'] ** 2 for x in G) / len(G):.5f}; corr(r, pw) {corr([x['pw'] for x in G], [x['r'] for x in G]):+.2f}")


if __name__ == "__main__":
    main()
