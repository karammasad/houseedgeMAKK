"""Phase 1 diagnostics for a two-lag size model (diagnosis only: validation truth is used for oracles and error
analysis, never for fitting or as a feature).

  python -m method2_sizes.twolag_probe --data data/main

1. Oracles on the validation TUNING half for the final model: true sizes; true binary size STATE (levels as learned);
   true win probability.
2. Where the P(1x|win) error sits: by condition, by observed win count, confident-wrong vs uncertain.
3. Missing-structure test on TRAIN histories: for machines whose single size lag k* is confidently identified,
   does splitting each k*-cell by "big win at a second lag j" add likelihood beyond a within-machine shuffle?
   Positive excess = a genuine two-lag interaction the current model cannot represent.
"""
import argparse, csv, math, os, random
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import folds, points, COLS, CONDITIONS
from method2_sizes.model import Model
from method2_sizes.round2 import FINAL


def sizes_of(p):
    pw = 1 - p[0]
    return pw, ([x / pw for x in p[1:]] if pw > 1e-12 else [.25] * 4)


def bern_dev(c):
    """2 x log-likelihood of counts [n0, n1] at their MLE."""
    n0, n1 = c; n = n0 + n1
    return 2 * sum(x * math.log(x / n) for x in (n0, n1) if x > 0) if n else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub)
    m = Model(**FINAL).fit(pub); g = m.sizes
    pred = m.predict(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    vq, vs = load_split(pub, "validation"); qm = {r["query_id"]: r for r in vq}

    def join(pw, s):
        return [1 - pw] + [pw * x for x in s]
    o_size, o_state, o_win = {}, {}, {}
    rows = []
    for k, p in pred.items():
        r = qm[k]; t = r["machine_type"]; pw, s = sizes_of(p); tw, ts = sizes_of(T[k])
        o_size[k] = join(pw, ts); o_win[k] = join(tw, s)
        lo, hi = g.levels[t]; st = hi if abs(ts[0] - hi) < abs(ts[0] - lo) else lo
        o_state[k] = join(pw, [st] + [(1 - st) * x for x in g.split[t]])
        if r["episode_id"] in tune:
            post = g.posterior_detail(r, vs[r["machine_id"]]); conf = max(f + tt for _, f, tt in post)
            nw = sum(1 for v in vs[r["machine_id"]] if v > 0)
            rows.append(dict(c=cond[k], err=abs(s[0] - ts[0]), conf=conf, nw=nw, tw=tw,
                             wrong=(abs(s[0] - ts[0]) > (hi - lo) / 2)))
    print("== 1. Oracles (validation tuning half, final model = 57.901 tuning) ==")
    for name, v in (("final model", pred), ("true sizes", o_size), ("true size STATE (learned levels)", o_state),
                    ("true win probability", o_win)):
        tp, tr = points(pub, v, tune)
        print(f"  {name:34s} {tp:7.3f}   " + "  ".join(f"{c[:6]} {tr[c]:.4f}" for c in CONDITIONS))

    print("\n== 2. |P(1x|win) error| on tuning queries (weighted by true win prob^2 in the score) ==")
    by = defaultdict(list)
    for x in rows:
        by[x["c"]].append(x)
    for c in CONDITIONS:
        v = by[c]
        print(f"  {c:14s} mean |a err| {sum(x['err'] for x in v) / len(v):.3f}   state wrong {sum(x['wrong'] for x in v) / len(v):.0%}"
              f"   confident(>.9) & wrong {sum(x['wrong'] and x['conf'] > .9 for x in v) / len(v):.0%}")
    for lo_, hi_ in ((0, 20), (20, 40), (40, 80), (80, 1000)):
        v = [x for x in rows if lo_ <= x["nw"] < hi_]
        if v:
            print(f"  wins {lo_:3d}-{hi_ if hi_ < 1000 else '+':>4}: n={len(v):3d}  mean |a err| {sum(x['err'] for x in v) / len(v):.3f}"
                  f"   state wrong {sum(x['wrong'] for x in v) / len(v):.0%}")

    print("\n== 3. Missing two-lag structure? (train, machines with confident single size lag) ==")
    q, seq = load_split(pub, "train"); rng = random.Random(7)
    gain = defaultdict(float); null = defaultdict(float); used = 0
    for r in q:
        s = seq[r["machine_id"]]; t = r["machine_type"]
        if t not in g.levels:
            continue
        post = g.posterior_detail(r, s); k, f, tt = max(post, key=lambda x: x[1] + x[2])
        if k == 0 or f + tt < .9:
            continue
        used += 1
        wins = [(tt_, int(s[tt_] == 1)) for tt_ in range(8, len(s)) if s[tt_] > 0 and s[tt_ - k] >= 0]
        cat = lambda x: int(x > 1)   # learned partition is big vs rest (.96)
        for j in range(1, 9):
            if j == k:
                continue
            obs = [(cat(s[t_ - k]), cat(s[t_ - j]), y) for t_, y in wins if s[t_ - j] >= 0]
            if len(obs) < 20:
                continue
            def dev(o):
                base = defaultdict(lambda: [0, 0]); split = defaultdict(lambda: [0, 0])
                for gk, gj, y in o:
                    base[gk][y] += 1; split[(gk, gj)][y] += 1
                return sum(map(bern_dev, split.values())) - sum(map(bern_dev, base.values()))
            gain[j - k] += dev(obs)   # keyed by offset j - k to pool machines with different k
            js = [gj for _, gj, _ in obs]; rng.shuffle(js)
            null[j - k] += dev([(gk, gj, y) for (gk, _, y), gj in zip(obs, js)])
    print(f"  machines used: {used}.  Deviance gain from a second lag (offset j - k*) vs shuffled second lag:")
    for off in sorted(gain, key=lambda o: -(gain[o] - null[o]))[:8]:
        print(f"    offset {off:+d}: gain {gain[off]:7.1f}  null {null[off]:7.1f}  excess {gain[off] - null[off]:+7.1f}")
    tot_g = sum(gain.values()); tot_n = sum(null.values())
    print(f"  all offsets: gain {tot_g:.1f}  null {tot_n:.1f}  excess {tot_g - tot_n:+.1f}  "
          f"(2 extra params per machine-offset: expected excess ~0 if no interaction)")


if __name__ == "__main__":
    main()
