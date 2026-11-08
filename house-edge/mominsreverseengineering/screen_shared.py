"""Training-only screening of the integrated shared-base structure, FLIPS ENABLED in both models. No validation, no audit.

V0 = v8 production gate (flip prior as shipped).  V1 = the same gate plus the learned "xsg" structure (shared base rate + size-state shift),
change-point inference intact (flip times only remap cell counts before the likelihood). Both read the same pseudo-queries.
Windows: PRE-FLIP targets at 45-58% of a long log and POST-FLIP targets at 80-92% (drift / shared-change machines have flipped by then; the prefix
contains the flip). Information settings relative to each target: n40, real short layout (40 consecutive + 24 random older), full history.
Targets are realized wins; paired squared-error differences are unbiased for differences against the true probability (absolute levels include the
Bernoulli floor). Condition labels (train conditions.csv) are used ONLY to stratify the report, never as a model input.
Caveat (screening, not a final test): the population parameters and structure weights of BOTH models were fit on all train histories, so both are
in-sample on these machines; the new structure has a few more fitted parameters. The frozen tuning evaluation is the real test.
Run: python3 screen_shared.py --data ../data/main
"""
import argparse, csv, os, pickle, random, time, zlib
from collections import defaultdict
from multiprocessing import get_context
from common.data import public_dir, load_split
import prefixdiag as D
import prefixdiag_flipfair as F

_S = D._S
WINDOWS = {"pre-flip 45-58%": (.45, .58, 2), "post-flip 80-92%": (.80, .92, 3)}
MASKS = ("n40", "short64", "full")


def job(args):
    r, = args; s = _S["seq"][r["machine_id"]]; L = len(s); rows = []
    for wname, (a, b, step) in WINDOWS.items():
        for t in range(int(a * L), int(b * L), step):
            y = int(s[t] > 0)
            for mask in MASKS:
                ss = D.masked(s, t, D.keep_set(mask, t, s, r["machine_id"]))
                rows.append(dict(ep=r["episode_id"], mid=r["machine_id"], type=r["machine_type"], cond=_S["cond"][r["query_id"]], win=wname, mask=mask, y=y,
                                 V0=F.v0_predict_pc(_S["gm0"], r, t, ss), V1=F.v0_predict_pc(_S["gm1"], r, t, ss)))
    return rows


def mean(v):
    return sum(v) / len(v)


def fmt(rs):
    d = [((r["V1"] - r["y"]) ** 2 - (r["V0"] - r["y"]) ** 2) * 1e4 for r in rs]; lo, hi = D.boot(d, [r["ep"] for r in rs])
    return f"{mean(d):+7.2f} [{lo:+7.2f},{hi:+7.2f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    pub = public_dir(a.data); t0 = time.time()
    from method4_ml.model import Model
    _S["gm0"] = Model(cache=True)._fit_gate(pub); _S["gm1"] = pickle.load(open("experiments/gm_sb.pkl", "rb"))
    q, seq = load_split(pub, "train"); _S["seq"] = seq
    _S["cond"] = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "train", "conditions.csv")))}
    long_rows = [r for r in q if sum(v < 0 for v in seq[r["machine_id"]]) < 0.1 * len(seq[r["machine_id"]])]
    with get_context("fork").Pool(8) as pool:
        out = pool.map(job, [(r,) for r in long_rows], chunksize=4)
    rows = [x for o in out for x in o]; print(f"{len(rows)} rows, {len({r['mid'] for r in rows})} long machines, {len({r['ep'] for r in rows})} episodes, {time.time() - t0:.0f}s")
    print("\nPAIRED squared-error difference V1 - V0 (x1e-4; negative = new structure better; 95% CI clustered by episode)")
    print(f"{'window':18s} {'info':8s} " + " | ".join(f"{k:>24s}" for k in ("ALL", "stable", "drift", "shared_change")))
    for w in WINDOWS:
        for m in MASKS:
            line = f"{w:18s} {m:8s} "; cells = []
            for c in (None, "stable", "drift", "shared_change"):
                rs = [r for r in rows if r["win"] == w and r["mask"] == m and (c is None or r["cond"] == c)]; cells.append(fmt(rs))
            print(line + " | ".join(f"{x:>24s}" for x in cells))
    print("\nBy machine type (ALL conditions):")
    for w in WINDOWS:
        for m in MASKS:
            print(f"  {w:18s} {m:8s} " + "  ".join(f"{ty} {fmt([r for r in rows if r['win'] == w and r['mask'] == m and r['type'] == ty])}" for ty in ("balanced", "jackpot", "steady")))


if __name__ == "__main__":
    main()
