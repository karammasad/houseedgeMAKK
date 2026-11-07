"""Experiment 13: joint win x size hidden-state marginalization vs win-only marginalization (short logs).

  python -m method2_sizes.jointfilter_eval PARAMS.json CACHE.json --data data/main

Increments are added to FINAL's cached forecast: V' = V_FINAL + (V_on - V_off), V_on/V_off = the filter's posterior-mean
5-vector with/without the unknown-input ticks (clipped at 0, renormalized). Structure prior learned on TRAIN only.
Reports five-category error, its win/size split (analysis vs validation truth), and the tuning-half score.
"""
import argparse, csv, json, os, sys, time
from common.data import load_split, public_dir
from common.evaluate import folds, points, COLS, CONDITIONS
from adi.model import is_short
from method2_sizes.missfilter import MissFilter
from method2_sizes.jointfilter import JointFilter
from method2_sizes.decompose import parts


def apply(base, inc):
    v = [max(1e-6, x + d) for x, d in zip(base, inc)]; z = sum(v); return [x / z for x in v]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("params"); ap.add_argument("cache"); ap.add_argument("--data", default="data/main")
    a = ap.parse_args(); pub = public_dir(a.data); tune, _ = folds(pub)
    par = json.load(open(a.params)); C = json.load(open(a.cache))["final"]["validation"]
    tq, ts = load_split(pub, "train"); mf = MissFilter(par["signal_priors"], G=12).fit(tq, ts); jf = JointFilter(mf, par, G=12)
    print(f"kept hypotheses {len(jf.hyps)}: {[(n, round(p, 3), 'k=' + str(jf.klag(n[0]))) for n, p in jf.hyps][:6]}; partition group {jf.grp}")
    vq, vs = load_split(pub, "validation")
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(os.path.join(pub, "validation", "targets.csv")))}
    V = {"FINAL": dict(C), "win-only inc": dict(C), "joint inc": dict(C), "joint WIN part": dict(C), "joint SIZE part": dict(C)}; t0 = time.time(); n = skipped = 0
    for r in vq:
        s = vs[r["machine_id"]]
        if not is_short(s):
            continue
        k = r["query_id"]; base = [x / sum(C[k]) for x in C[k]]
        on, off = jf.run(r, s, True), jf.run(r, s, False)
        if on is None or off is None:
            skipped += 1; continue
        V["joint inc"][k] = apply(base, [x - y for x, y in zip(on, off)])
        # split the joint increment: P(win) change only (FINAL's size split kept) / size-split change only (FINAL's P(win) kept)
        pw0 = 1 - base[0]; pw_j = min(.999, max(.001, pw0 + (1 - on[0]) - (1 - off[0]))); sz0 = [x / pw0 for x in base[1:]]
        a_on, a_off = on[1] / (1 - on[0]), off[1] / (1 - off[0]); big = [x / (1 - a_on) for x in on[2:]]
        a_j = min(.99, max(.01, sz0[0] + a_on - a_off)); bg = [x / sum(sz0[1:]) for x in sz0[1:]]
        V["joint WIN part"][k] = [1 - pw_j] + [pw_j * x for x in sz0]
        V["joint SIZE part"][k] = [1 - pw0, pw0 * a_j] + [pw0 * (1 - a_j) * x for x in bg]
        pw = mf.pwin(r, s, True) - mf.pwin(r, s, False); sz = [x / (1 - base[0]) for x in base[1:]]
        pw2 = min(.999, max(.001, 1 - base[0] + pw)); V["win-only inc"][k] = [1 - pw2] + [pw2 * x for x in sz]; n += 1
    print(f"filters on {n} validation short logs ({skipped} skipped, cutoff lags unobserved): {time.time() - t0:.0f}s")
    sel = [r for r in vq if is_short(vs[r["machine_id"]]) and r["episode_id"] in tune]
    print(f"\nfive-category error on tuning-half short queries (n={len(sel)}), analysis only:")
    print(f"  {'variant':14s} {'total':>9s} {'WIN part':>9s} {'SIZE part':>10s}   tuning score  short ratio")
    for name, P in V.items():
        w = z = t = 0.0
        for r in sel:
            k = r["query_id"]; ww, zz, cc, *_ = parts(P[k], T[k]); w += ww; z += zz + cc; t += sum((x - y) ** 2 for x, y in zip(P[k], T[k]))
        tp, tr = points(pub, P, tune)
        print(f"  {name:14s} {t / len(sel):9.5f} {w / len(sel):9.5f} {z / len(sel):10.5f}   {tp:8.3f}      {tr['short_history']:.4f}")


if __name__ == "__main__":
    main()
