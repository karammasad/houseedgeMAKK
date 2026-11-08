"""Win-informed size direction: ONE training-only, episode-grouped, held-out diagnostic. No validation file, no audit, no targets.

Pseudo-queries: long TRAIN machines, target ticks u in [45%, 58%) of the log (flip-free), prefixes relative to u: n40, the real short layout, full history;
inference uses only ticks before u. Targets are the REALIZED payouts at u. Baseline = our existing shared-base model (gate with the xsg structure, size model,
size menu): p_win from the gate (identical for every variant), sizes from GateSizes with win_dir_lam = 0 (exactly the shipped model), snapped to the menu.
Candidate = the same with win_dir_lam = lambda (win evidence updates the size direction). Control = the same evidence computed from ANOTHER machine of the same
type and a different episode (its win/category history at the same positions), breaking the win-state link while keeping the size evidence.
Metrics: (1) size-state inference: held-out log-loss of the realized size class on winning ticks; (2) final five-category squared error on all ticks.
lambda is chosen from {-0.5, 0, .25, .5, 1} on one half of the episodes and EVALUATED on the other (2-fold, grouped by episode); negative lambda would mean the
sign link is reversed (so the sign is learned, not assumed).
PRE-STATED DECISION RULE (fixed before running):
  PROCEED to one official tuning-half run only if, on short64 and n40, the cross-fitted size log-loss improves with an episode-clustered 95% CI below 0,
  the shuffled-history control shows no improvement (its CI includes 0 or is worse), and the final five-category squared error does not worsen
  (upper 95% bound of the change <= +0.5e-4). Otherwise STOP.
Run: python3 wd_diag.py --data ../data/main
"""
import argparse, math, random, time, zlib
from collections import defaultdict
from multiprocessing import get_context
from common.data import public_dir, load_split
import hierarchy as H
import prefixdiag as D
import prefixdiag_flipfair as F
from method4_ml.shared_base import win_direction_evidence
from method4_ml.size_menu import snap

LAMS = (-0.5, 0.0, 0.25, 0.5, 1.0); MASKS = ("n40", "short64", "full"); SIZES = (1, 2, 5, 20)
_S = D._S


def partner_of(r, longs):
    c = [x for x in longs if x["machine_type"] == r["machine_type"] and x["episode_id"] != r["episode_id"]]
    return c[zlib.crc32(r["machine_id"].encode()) % len(c)]


def job(args):
    r, = args; seq = _S["seq"]; s = seq[r["machine_id"]]; L = len(s); gm, g, menu = _S["gm"], _S["g"], _S["menu"]
    pr = partner_of(r, _S["longs"]); ps = seq[pr["machine_id"]]; rows = []
    memo = {}
    def real(ctx, q, seq_m, dl):
        k = ("real", id(seq_m), dl)
        if k not in memo: memo[k] = win_direction_evidence(ctx, q, seq_m, dl)
        return memo[k]
    def shuffled(ctx, q, seq_m, dl):
        k = ("shuf", id(seq_m), dl)
        if k not in memo: memo[k] = win_direction_evidence(ctx, q, _S["pss"][0], dl)
        return memo[k]
    for u in range(int(.45 * L), int(.58 * L), 2):
        y = s[u]; onehot = [int(y == 0)] + [int(y == c) for c in SIZES]
        for mask in MASKS:
            keep = D.keep_set(mask, u, s, r["machine_id"]); ss = D.masked(s, u, keep); pss = D.masked(ps, u, keep); _S["pss"] = [pss]; memo.clear()
            pw = min(max(F.v0_predict_pc(gm, r, u, ss), 1e-6), 1 - 1e-6)
            rec = dict(ep=r["episode_id"], type=r["machine_type"], mask=mask, win=int(y > 0), cls=(SIZES.index(y) if y > 0 else -1))
            for tag, fn in (("real", real), ("shuf", shuffled)):
                g.win_dir_fn = fn
                for lam in LAMS:
                    if tag == "shuf" and lam == 0.0:
                        continue
                    g.win_dir_lam = lam; sz = snap(g.predict(r, ss), menu.get(r["machine_type"]))
                    p5 = [1 - pw] + [pw * x for x in sz]
                    rec[(tag, lam, "brier")] = sum((a - b) ** 2 for a, b in zip(p5, onehot))
                    if y > 0:
                        rec[(tag, lam, "sizell")] = -math.log(max(sz[SIZES.index(y)], 1e-9))
            g.win_dir_lam = 0.0; rows.append(rec)
    return rows


def mean(v):
    return sum(v) / len(v)


def val(r, tag, lam, kind):
    """lambda = 0 is the baseline under BOTH tags (no evidence is used), so the control reuses the real run's lambda = 0 value."""
    return r[("real" if lam == 0.0 else tag, lam, kind)]


def cf(d, eps, reps=1500, seed=5):
    by = defaultdict(list)
    for x, e in zip(d, eps):
        by[e].append(x)
    keys = sorted(by); rng = random.Random(seed); ms = []; S = {k: (sum(v), len(v)) for k, v in by.items()}
    for _ in range(reps):
        ch = [S[rng.choice(keys)] for _ in keys]; ms.append(sum(a for a, _ in ch) / sum(b for _, b in ch))
    ms.sort(); return ms[int(.025 * reps)], ms[int(.975 * reps)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args(); pub = public_dir(a.data); t0 = time.time()
    from method4_ml.model import Model
    m = Model(cache=True, shared_base=True).fit(pub)                       # cached fit: gate with xsg, size model, menu (stackers unused here)
    g = m.size_models["gate"]; g.win_dir_ctx = m.gm
    q, seq = load_split(pub, "train"); _S.update(seq=seq, gm=m.gm, g=g, menu=m.menu)
    longs = [r for r in q if sum(v < 0 for v in seq[r["machine_id"]]) < 0.1 * len(seq[r["machine_id"]])]; _S["longs"] = longs
    with get_context("fork").Pool(8) as pool:
        out = pool.map(job, [(r,) for r in longs], chunksize=4)
    rows = [x for o in out for x in o]
    import pickle; pickle.dump(rows, open("experiments/wd_rows.pkl", "wb"))
    print(f"{len(rows)} (target, mask) rows from {len(longs)} long machines, {len({r['ep'] for r in rows})} episodes, {time.time() - t0:.0f}s", flush=True)
    half = lambda r: zlib.crc32(r["ep"].encode()) % 2
    print("\nin-sample mean size log-loss on winning ticks by lambda (informational; REAL evidence | SHUFFLED control):")
    for tag in ("real", "shuf"):
        print(f"  {tag:5s} " + "  ".join(f"lam {l:+.2f}: {mean([r[(tag, l, 'sizell')] for r in rows if r['win'] and (tag, l, 'sizell') in r]):.5f}" for l in LAMS if not (tag == 'shuf' and l == 0.0)))
    def pick(tag, hrows):                              # lambda minimising size log-loss on the fitting half (0 = baseline)
        sc = {l: mean([r[("real" if l == 0.0 else tag, l, "sizell")] for r in hrows if r["win"]]) for l in LAMS}
        return min(sc, key=sc.get)
    verdicts = {}
    for tag in ("real", "shuf"):
        lam_of = {h: pick(tag, [r for r in rows if half(r) != h]) for h in (0, 1)}      # fit on the OTHER half
        print(f"\n[{tag.upper()}] cross-fitted lambda: evaluate half 0 with {lam_of[0]:+.2f}, half 1 with {lam_of[1]:+.2f}")
        print(f"{'':8s}{'info':8s} {'n wins':>7s} {'size log-loss base':>19s} {'change (cand - base)':>26s} | {'5-class sq err change x1e-4':>30s}")
        for mask in MASKS:
            rs = [r for r in rows if r["mask"] == mask]; w = [r for r in rs if r["win"]]
            def lamr(r): return lam_of[half(r)]
            dl = [val(r, tag, lamr(r), "sizell") - r[("real", 0.0, "sizell")] for r in w]; el = [r["ep"] for r in w]
            db = [(val(r, tag, lamr(r), "brier") - r[("real", 0.0, "brier")]) * 1e4 for r in rs]; eb = [r["ep"] for r in rs]
            lo, hi = cf(dl, el); blo, bhi = cf(db, eb)
            print(f"{'':8s}{mask:8s} {len(w):7d} {mean([r[('real', 0.0, 'sizell')] for r in w]):19.5f} {mean(dl):+10.5f} [{lo:+.5f},{hi:+.5f}] | {mean(db):+8.2f} [{blo:+7.2f},{bhi:+7.2f}]")
            verdicts[(tag, mask)] = (lo, hi, blo, bhi, mean(dl), mean(db))
        for ty in ("balanced", "jackpot", "steady"):
            w = [r for r in rows if r["win"] and r["mask"] == "short64" and r["type"] == ty]
            dl = [val(r, tag, lam_of[half(r)], "sizell") - r[("real", 0.0, "sizell")] for r in w]; lo, hi = cf(dl, [r["ep"] for r in w])
            print(f"{'':8s}short64 {ty:9s} n={len(w):4d} size log-loss change {mean(dl):+.5f} [{lo:+.5f},{hi:+.5f}]")
    ok = all(verdicts[("real", mk)][1] < 0 for mk in ("n40", "short64")) and all(verdicts[("shuf", mk)][0] <= 0 <= verdicts[("shuf", mk)][1] or verdicts[("shuf", mk)][0] > 0 for mk in ("n40", "short64")) \
         and all(verdicts[("real", mk)][3] <= 0.5 for mk in MASKS)
    print(f"\nPRE-STATED RULE -> {'PROCEED to ONE official tuning-half run' if ok else 'STOP (not met)'}   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
