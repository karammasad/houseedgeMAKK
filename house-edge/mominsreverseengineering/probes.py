"""Generator-constraint probes. DIAGNOSIS ONLY: reads validation targets of the TUNING half (never the audit half), never fit/predict.

  python3 probes.py --data data/main            # all probes, tuning half only
  python3 probes.py --data data/main --probe sizes

Each probe tests one "the generator is more restricted than our model assumes" hypothesis and prints a verdict line.
They are written to run unchanged on the round-two kit, so we find out in minutes which restrictions still hold.
A restriction only becomes a model change if it can be LEARNED from train/history data inside fit() (like the size menu).

Probes:
  sizes    : P(size | win) takes few distinct vectors per machine type? (v8's menu says exactly 2)
  winrate  : true P(win) clusters at few levels per type x signal (discrete menu) or is continuous?
  signal   : does `signal` predict the hidden size state or the win level beyond a small shift?
  variance : how much of true P(win) / P(1x|win) is explained by type, signal, episode (ICC)?
"""
import argparse, csv, math, os
from collections import defaultdict
from common.data import public_dir
from common.evaluate import folds


def tuning_truth(pub):
    tune, _ = folds(pub)                                    # tuning half ONLY. The audit half is never read here.
    q = {r["query_id"]: r for r in csv.DictReader(open(os.path.join(pub, "validation", "queries.csv")))}
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "validation", "conditions.csv")))}
    rows = []
    for t in csv.DictReader(open(os.path.join(pub, "validation", "targets.csv"))):
        if t["episode_id"] not in tune:
            continue
        p = [float(t[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]
        w = 1 - p[0]
        r = dict(q[t["query_id"]], p=p, pw=w, size=[x / w for x in p[1:]], cond=cond[t["query_id"]])
        r["signal"] = int(r["signal"])
        rows.append(r)
    return rows


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def sd(x):
    m = mean(x); return math.sqrt(mean([(v - m) ** 2 for v in x])) if len(x) > 1 else float("nan")


def probe_sizes(rows):
    print("== sizes: distinct P(size | win) vectors per type (rounded to 3 decimals)")
    for ty in sorted({r["machine_type"] for r in rows}):
        vs = defaultdict(int)
        for r in rows:
            if r["machine_type"] == ty:
                vs[tuple(round(x, 3) for x in r["size"])] += 1
        top = sorted(vs.items(), key=lambda kv: -kv[1])
        covered = sum(c for _, c in top[:2]) / sum(vs.values())
        print(f"  {ty:9s} {len(vs):3d} distinct vectors; top 2 cover {covered:.0%}; top: " + "; ".join(f"{list(v)} x{c}" for v, c in top[:2]))
        verdict = "MENU HOLDS (<=2 vectors)" if len(vs) <= 2 else "menu BROKEN or noisy: re-check size_menu"
        print(f"    -> {verdict}")


def probe_winrate(rows):
    print("== winrate: true P(win) by type x signal (continuous vs few levels)")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["machine_type"], r["signal"])].append(r["pw"])
    for k in sorted(cells):
        v = sorted(cells[k]); distinct = len({round(x, 3) for x in v})
        print(f"  {k[0]:9s} signal {k[1]:+d}: n={len(v):3d} mean {mean(v):.3f} sd {sd(v):.3f} distinct {distinct:3d} (n={len(v)}) range {v[0]:.3f}-{v[-1]:.3f}")
    print("  -> discrete menu only if `distinct` is far below n within each cell")


def probe_signal(rows):
    print("== signal: does signal predict the size state (nearer V1 or V2) or the win level?")
    byty = defaultdict(list)
    for r in rows:
        byty[r["machine_type"]].append(r)
    for ty, rs in sorted(byty.items()):
        x = [r["size"][0] for r in rs]                      # P(1x | win): the dimension that carries the two-state size gate
        mid = (min(x) + max(x)) / 2
        tab = defaultdict(lambda: [0, 0])
        for r in rs:
            tab[r["signal"]][r["size"][0] > mid] += 1
        print(f"  {ty:9s} hi-1x state share by signal: " + ", ".join(f"{s:+d}: {tab[s][1]}/{sum(tab[s])}" for s in sorted(tab)))
        pw = {s: mean([r["pw"] for r in rs if r["signal"] == s]) for s in sorted({r["signal"] for r in rs})}
        print(f"            mean P(win) by signal: " + ", ".join(f"{s:+d}: {v:.3f}" for s, v in pw.items()))
    print("  -> a hidden-parameter signal would show a state share near 0 or 1 for some signal, not ~even splits")


def probe_variance(rows):
    print("== variance: share of true P(win) variance explained by groups (eta^2; episode is the ICC proxy)")
    tot = sd([r["pw"] for r in rows]) ** 2
    for name, key in (("type", lambda r: r["machine_type"]), ("signal", lambda r: r["signal"]),
                      ("type x signal", lambda r: (r["machine_type"], r["signal"])), ("episode", lambda r: r["episode_id"]),
                      ("condition", lambda r: r["cond"])):
        g = defaultdict(list)
        for r in rows:
            g[key(r)].append(r["pw"])
        grand = mean([r["pw"] for r in rows])
        between = sum(len(v) * (mean(v) - grand) ** 2 for v in g.values()) / len(rows)
        print(f"  {name:14s} groups {len(g):3d}  eta^2 {between / tot:.3f}")
    print("  -> episode eta^2 is inflated by small groups (8 machines each); compare with its chance level ~ groups/n")


def probe_cells(rows):
    print("== cells: true P(win) is bimodal (gate on/off). Within each mode, per type: mean, sd, and by-signal means")
    for ty in sorted({r["machine_type"] for r in rows}):
        rs = [r for r in rows if r["machine_type"] == ty]
        lg = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))
        x = sorted(lg(r["pw"]) for r in rs); c = [x[len(x) // 4], x[3 * len(x) // 4]]
        for _ in range(30):                                  # 1-D two-means on the logit
            lo = [v for v in x if abs(v - c[0]) <= abs(v - c[1])]; hi = [v for v in x if abs(v - c[0]) > abs(v - c[1])]
            c = [mean(lo), mean(hi)]
        thr = (c[0] + c[1]) / 2
        for name, sel in (("off", lambda r: lg(r["pw"]) <= thr), ("on", lambda r: lg(r["pw"]) > thr)):
            g = [r for r in rs if sel(r)]
            ps = [r["pw"] for r in g]
            bys = ", ".join(f"{s:+d}: {mean([r['pw'] for r in g if r['signal'] == s]):.3f}" for s in sorted({r['signal'] for r in g}))
            print(f"  {ty:9s} {name:3s} n={len(g):3d} mean P {mean(ps):.3f} sd {sd(ps):.3f} (logit sd {sd([lg(p) for p in ps]):.2f})  by signal -> {bys}")
    print("  -> a discrete rate menu would show tiny sd within a mode; Beta-like noise shows sd ~0.08-0.16")


def probe_shape(rows):
    """Beta vs logit-normal for the true rates inside each (type, mode, signal) cell: same number of parameters (2) each,
    compared by the log-likelihood of the P(win) values on the probability scale (logit-normal gets its Jacobian)."""
    try:
        from scipy.stats import beta as Beta
    except ImportError:
        print("== shape: scipy missing, skipped"); return
    print("== shape: Beta vs logit-normal likelihood of true cell rates (diagnosis; positive = logit-normal better)")
    lg = lambda p: math.log(p / (1 - p))
    total = 0.0; n_tot = 0
    for ty in sorted({r["machine_type"] for r in rows}):
        rs = [r for r in rows if r["machine_type"] == ty]
        x = sorted(lg(min(max(r["pw"], 1e-6), 1 - 1e-6)) for r in rs); c = [x[len(x) // 4], x[3 * len(x) // 4]]
        for _ in range(30):
            lo = [v for v in x if abs(v - c[0]) <= abs(v - c[1])]; hi = [v for v in x if abs(v - c[0]) > abs(v - c[1])]
            c = [mean(lo), mean(hi)]
        thr = (c[0] + c[1]) / 2
        for mode in ("off", "on"):
            for s in (-1, 0, 1):
                ps = [min(max(r["pw"], 1e-4), 1 - 1e-4) for r in rs if r["signal"] == s and (lg(min(max(r["pw"], 1e-6), 1 - 1e-6)) > thr) == (mode == "on")]
                if len(ps) < 8:
                    continue
                z = [lg(p) for p in ps]; m, v = mean(z), max(sd(z) ** 2 * (len(z) - 1) / len(z), 1e-9)
                ll_n = sum(-math.log(p * (1 - p)) - 0.5 * math.log(2 * math.pi * v) - (lg(p) - m) ** 2 / (2 * v) for p in ps)
                a, b, _, _ = Beta.fit(ps, floc=0, fscale=1)
                ll_b = sum(Beta.logpdf(p, a, b) for p in ps)
                total += ll_n - ll_b; n_tot += len(ps)
                print(f"  {ty:9s} {mode:3s} signal {s:+d} n={len(ps):2d}  loglik logit-normal {ll_n:8.2f}  Beta {ll_b:8.2f}  diff {ll_n - ll_b:+6.2f}")
    print(f"  total over {n_tot} values: logit-normal minus Beta = {total:+.1f} log-lik units ({total / max(n_tot, 1):+.3f} per value)")
    print("  -> a few units is noise at this n; >10 would justify the logit-normal gate prior test")


def probe_state(rows):
    """Does the hidden size state move the WIN rate, and how much of the within-mode win-rate variance is that state?
    State = which of the type's two size vectors the query is on (low vs high P(1x|win)). Mode = off/on gate cell (2-means on logit).
    Upper bound for any model that knew the state: share of within-(type, mode) variance of P(win) explained by the state."""
    print("== state: size state vs win rate, within type x gate mode (truth, tuning half)")
    lg = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))
    cells = []                                        # (type, mode, state, pw, cond)
    for ty in sorted({r["machine_type"] for r in rows}):
        rs = [r for r in rows if r["machine_type"] == ty]
        x = sorted(lg(r["pw"]) for r in rs); c = [x[len(x) // 4], x[3 * len(x) // 4]]
        for _ in range(30):
            lo = [v for v in x if abs(v - c[0]) <= abs(v - c[1])]; hi = [v for v in x if abs(v - c[0]) > abs(v - c[1])]
            c = [mean(lo), mean(hi)]
        thr = (c[0] + c[1]) / 2
        s1 = [r["size"][0] for r in rs]; mid = (min(s1) + max(s1)) / 2
        for r in rs:
            cells.append((ty, "on" if lg(r["pw"]) > thr else "off", int(r["size"][0] > mid), r["pw"], r["cond"]))
    tot_w = tot_b = 0.0
    for ty in sorted({c[0] for c in cells}):
        for mode in ("off", "on"):
            g = [c for c in cells if c[0] == ty and c[1] == mode]
            m1 = [c[3] for c in g if c[2] == 1]; m0 = [c[3] for c in g if c[2] == 0]
            if len(m1) < 3 or len(m0) < 3:
                continue
            allp = [c[3] for c in g]; gm = mean(allp)
            within = sum((p - gm) ** 2 for p in allp)
            between = len(m1) * (mean(m1) - gm) ** 2 + len(m0) * (mean(m0) - gm) ** 2
            tot_w += within; tot_b += between
            print(f"  {ty:9s} {mode:3s} high-1x state: n={len(m1):2d} mean P(win) {mean(m1):.3f} | low-1x state: n={len(m0):2d} mean {mean(m0):.3f} | "
                  f"diff {mean(m1) - mean(m0):+.3f}  state explains {between / within:.0%} of within-mode variance")
    print(f"  pooled: state explains {tot_b / tot_w:.0%} of within-(type, mode) variance of true P(win)")
    for cond in ("short_history", "stable"):
        w = b = 0.0
        for ty in sorted({c[0] for c in cells}):
            for mode in ("off", "on"):
                g = [c for c in cells if c[0] == ty and c[1] == mode and c[4] == cond]
                m1 = [c[3] for c in g if c[2] == 1]; m0 = [c[3] for c in g if c[2] == 0]
                if len(m1) < 2 or len(m0) < 2:
                    continue
                allp = [c[3] for c in g]; gm = mean(allp)
                w += sum((p - gm) ** 2 for p in allp)
                b += len(m1) * (mean(m1) - gm) ** 2 + len(m0) * (mean(m0) - gm) ** 2
        print(f"  condition {cond:13s}: state explains {b / max(w, 1e-12):.0%} of within-mode variance (small n; indicative only)")
    print("  -> the state-share is the CEILING for what a model that perfectly knew the state could remove; sign shows the tie")


PROBES = {"sizes": probe_sizes, "winrate": probe_winrate, "signal": probe_signal, "variance": probe_variance, "cells": probe_cells,
          "shape": probe_shape, "state": probe_state}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--probe", choices=sorted(PROBES))
    a = ap.parse_args(); pub = public_dir(a.data)
    rows = tuning_truth(pub)
    print(f"tuning half: {len(rows)} queries, {len({r['episode_id'] for r in rows})} episodes (audit half untouched)")
    for name, fn in PROBES.items():
        if not a.probe or a.probe == name:
            fn(rows)


if __name__ == "__main__":
    main()
