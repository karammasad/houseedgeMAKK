"""Step 1 error decomposition (diagnosis only; reads validation targets, never inside fit/predict).

  python -m method2_sizes.decompose --data data/main --kwargs '<Model settings JSON>' [--method method2_sizes]

Exact split of each query's squared error, with p = [1-pw, pw*s] and truth [1-tw, tw*t]:
  sum_k (pw s_k - tw t_k)^2 = sum_k ((pw-tw) s_k + tw (s_k - t_k))^2
  WIN   = (pw-tw)^2 * (1 + |s|^2)              error if sizes were the model's but win prob were wrong
  SIZE  = tw^2 * |s - t|^2                      error from the size split, weighted by the true win prob
  CROSS = 2 (pw-tw) tw <s, s - t>               interaction (small, either sign)
Per condition, each part is reported in "loss-ratio units" (episode-averaged like the metric, divided by the
reference error) so they add up to the condition's loss ratio.
"""
import argparse, csv, importlib, json, math, os
from collections import defaultdict
from common.data import public_dir, load_split
from common.evaluate import COLS, CONDITIONS
from method1_gate.gate import structures, machine_tables, learn_structure_prior, logsumexp


def parts(p, t):
    pw, tw = 1 - p[0], 1 - t[0]
    s = [x / pw for x in p[1:]] if pw > 1e-12 else [.25] * 4
    ts = [x / tw for x in t[1:]] if tw > 1e-12 else s
    win = (pw - tw) ** 2 * (1 + sum(x * x for x in s))
    size = tw ** 2 * sum((a - b) ** 2 for a, b in zip(s, ts))
    cross = 2 * (pw - tw) * tw * sum(a * (a - b) for a, b in zip(s, ts))
    return win, size, cross, pw, tw, s, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/main"); ap.add_argument("--method", default="method2_sizes")
    ap.add_argument("--kwargs", default="{}"); ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(); pub = public_dir(a.data)
    model = importlib.import_module(f"{a.method}.model").Model(**json.loads(a.kwargs)).fit(pub)
    pred = model.predict(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    ref = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("reference.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    q, seq = load_split(pub, "validation"); qm = {r["query_id"]: r for r in q}

    # metric-consistent aggregation: mean within episode, mean over episodes, / reference error
    acc = defaultdict(lambda: defaultdict(lambda: [0.0] * 5))
    rows = []
    for k, p in pred.items():
        s = sum(p); p = [x / s for x in p]
        w, z, c, pw, tw, sm, ts = parts(p, T[k]); tot = sum((x - y) ** 2 for x, y in zip(p, T[k]))
        rerr = sum((x - y) ** 2 for x, y in zip(ref[k], T[k]))
        e = acc[cond[k]][qm[k]["episode_id"]]
        for i, v in enumerate((w, z, c, tot, rerr)):
            e[i] += v; e[i] = e[i]
        e.append(1)
        rows.append((tot, k, w, z, c, pw, tw, sm, ts))
    print("== Loss ratio split per condition (WIN + SIZE + CROSS = ratio) ==")
    tw_ = [0.0] * 3; R = 0
    for cnd in CONDITIONS:
        eps = acc[cnd]; vals = [0.0] * 5
        for e in eps.values():
            n = len(e) - 5
            for i in range(5):
                vals[i] += e[i] / n / len(eps)
        ratio = [v / vals[4] for v in vals[:4]]
        print(f"  {cnd:14s} ratio {ratio[3]:.4f} = WIN {ratio[0]:.4f} + SIZE {ratio[1]:.4f} + CROSS {ratio[2]:+.4f}"
              f"   (win share {ratio[0] / ratio[3]:.0%})")
        for i in range(3):
            tw_[i] += ratio[i] / 4
        R += ratio[3] / 4
    print(f"  {'ALL (mean)':14s} R {R:.4f} = WIN {tw_[0]:.4f} + SIZE {tw_[1]:.4f} + CROSS {tw_[2]:+.4f}")

    # worst queries with what the model believed
    trn_q, trn_seq = load_split(pub, "train"); ST = structures(8, 2)
    sprior = learn_structure_prior(trn_q, trn_seq, ST)
    rows.sort(reverse=True)
    print(f"\n== {a.top} worst queries ==")
    print("  sqerr  WIN   SIZE  cond           type      pw->true   MAP win structure (post)  P(flip)  "
          "wins(last40) obs  P(1x|w) model->true")
    kinds = defaultdict(int)
    for tot, k, w, z, c, pw, tw, sm, ts in rows[:a.top]:
        r = qm[k]; s = seq[r["machine_id"]]; cut = len(s)
        taus = list(range(40, cut - 2, 2))
        lt = [math.log(.5 / len(taus))] * len(taus) + [math.log(.5)]
        tb = machine_tables(s, cut, taus, ST, .5, .5)
        lsum = [math.log(sprior[i]) + logsumexp([ll + l for (ll, _), l in zip(rows_, lt)]) for i, rows_ in enumerate(tb)]
        z_ = logsumexp(lsum); post = [math.exp(x - z_) for x in lsum]; i = max(range(len(post)), key=post.__getitem__)
        st = (ST + [()])[i]
        flip = math.exp(logsumexp([ll + l for (ll, _), l in zip(tb[i][:-1], lt[:-1])]) -
                        logsumexp([ll + l for (ll, _), l in zip(tb[i], lt)]))
        nobs = sum(v >= 0 for v in s); w40 = sum(v > 0 for v in s[-40:])
        kind = "C-misid/flip" if post[i] > .9 and abs(pw - tw) > .3 else ("A-uncertain gate" if w > z else "B-sizes")
        kinds[kind] += tot
        print(f"  {tot:.3f}  {w:.3f} {z:.3f}  {cond[k]:14s} {r['machine_type']:9s} {pw:.2f}->{tw:.2f}   "
              f"{str(st):10s} ({post[i]:.2f})          {flip:.2f}    {w40:2d}          {nobs:3d}  "
              f"{sm[0]:.2f}->{ts[0]:.2f}   [{kind}]")
    print("\nWorst-20 error mass by kind:", {k: round(v, 3) for k, v in kinds.items()})


if __name__ == "__main__":
    main()
