"""Diagnostic (analysis only; reads validation targets/conditions): where does the remaining WIN-probability
error come from?  Per condition, splits squared win error into
  - big misses  (|ours - true| > 0.3: wrong gate cell / wrong structure / missed flip)
  - small misses (rate calibration within the right cell)
and checks whether truth sits between our cells' levels (suggesting a third state or gradual drift).
Usage: python -m adi.win_diag --data data/main --kwargs '{...}'
"""
import argparse, csv, json, os
from collections import defaultdict
from common.data import public_dir, load_split
from adi.model import Model
from adi import gate as G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--kwargs", default="{}")
    a = ap.parse_args(); pub = public_dir(a.data)
    m = Model(**json.loads(a.kwargs)).fit(pub)
    q, seq = load_split(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    truth = {r["query_id"]: 1 - float(r["p_0"]) for r in rd("targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    agg = defaultdict(lambda: defaultdict(float)); mids = defaultdict(int); n = defaultdict(int)
    for r in q:
        s = seq[r["machine_id"]]; cfg = m.cfg
        from adi.model import is_short
        if m.p_change_short is not None and is_short(s):
            cfg = dict(cfg, p_change=m.p_change_short)
        pw = G.win_probability(r, s, m.ST, m.sprior, cellprior=m._cp(r), **cfg)
        tw = truth[r["query_id"]]; c = cond[r["query_id"]]; e = (pw - tw) ** 2
        n[c] += 1
        agg[c]["big" if abs(pw - tw) > .3 else "small"] += e
        agg[c]["n_big"] += abs(pw - tw) > .3
        cp = m._cp(r); on, off = cp[0] / (cp[0] + cp[1]), cp[2] / (cp[2] + cp[3])
        if off + .15 < tw < on - .15:
            mids[c] += 1
        agg[c]["over"] += e if pw > tw else 0
    print(f"{'condition':14s} {'n':>4s} {'big misses':>10s} {'share big':>9s} {'share small':>11s} {'truth mid-level':>15s}")
    for c in sorted(n):
        tot = agg[c]["big"] + agg[c]["small"]
        print(f"{c:14s} {n[c]:4d} {int(agg[c]['n_big']):10d} {agg[c]['big'] / tot:9.0%} {agg[c]['small'] / tot:11.0%} {mids[c]:15d}")


if __name__ == "__main__":
    main()
