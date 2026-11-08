"""Round 2 adaptation: one command, data-selected configuration, v8 fallback always available.  Run from mominsreverseengineering/.

  python3 adapt.py --data data/adaptation --out build/adapt            # select + final (full pipeline, test CSV, kit checker)
  python3 adapt.py --data data/adaptation --out build/adapt --stage select   # win-side selection only (minutes), writes chosen.json
  python3 adapt.py --data data/adaptation --out build/adapt --stage final    # reuse chosen.json

Pipeline (nothing here assumes Round 1's lag pairs, size-state count, shift or flip frequency; all are re-estimated):
  1. kit check: metadata round / max_points, splits present, validation answers present (selection needs them).
  2. fit v8's gate on the new TRAIN data (structure prior, per-type cell priors, signal slopes, short-log flip prior).
  3. candidate "sb" = the same gate + the shared-base-rate + size-shift structure ("xsg"): pairs are DERIVED from the fitted structure prior, population
     parameters and structure weights refit with convergence checks (shared_base.py). Parameters sitting on a bound / non-converged fits are printed.
  4. SELECT on the TUNING half only (validation true probabilities): win-side squared error per condition, normalised by the kit's frozen reference,
     base vs sb. sb is chosen only if its tuning-half mean ratio is lower by at least MARGIN and no condition is worse by more than WORSE.
     The AUDIT half is looked at ONCE, after the choice is frozen, as a guard: if sb is worse than base on audit it is dropped (v8 fallback).
  5. FINAL: the full v8 pipeline (gate + sizes + menu + stackers) with the chosen settings; official metric on the new validation (scale from this
     kit's metadata), test predictions, kit checker. chosen.json holds the kwargs for method4_ml.make_entry.
Fallback: --force base skips the candidate entirely (plain v8).
"""
import argparse, copy, csv, json, math, os, subprocess, sys, time
from collections import defaultdict
from common.data import public_dir, load_split, write_predictions
from common.evaluate import folds, points, CONDITIONS
from adi import gate as G
from method4_ml.model import Model
from method4_ml.pooled import pooled_split
from method4_ml.shared_base import attach_shared_base

HERE = os.path.dirname(os.path.abspath(__file__))
MARGIN, WORSE = 0.02, 0.10      # sb must beat base by >= 2% of the mean ratio on the tuning half; no condition worse by > 10%


def kit_check(pub):
    meta = json.load(open(os.path.join(pub, "metadata.json")))
    for sp in ("train", "validation", "test"):
        if not os.path.exists(os.path.join(pub, sp, "queries.csv")):
            sys.exit(f"missing {sp}/queries.csv in {pub}")
    have = os.path.exists(os.path.join(pub, "validation", "targets.csv"))
    print(f"kit: round={meta.get('round')} max_points={meta.get('max_points')} season={meta.get('season_id')} validation answers: {have}")
    if meta.get("round") not in ("adaptation",):
        print("  NOTE: metadata round is not 'adaptation' (rehearsal or synthetic kit)")
    return meta, have


def win_ratios(pub, gm, episodes):
    """Per-condition win-side error (vs the TRUE P(win) = 1 - p_0) of the gate, normalised by the frozen reference's, over the given episodes."""
    q, seq = load_split(pub, "validation"); rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}; ref = {r["query_id"]: 1 - float(r["p_0"]) for r in rd("reference.csv")}
    truth = {t["query_id"]: (t["episode_id"], 1 - float(t["p_0"])) for t in rd("targets.csv")}
    e = defaultdict(lambda: defaultdict(list)); b = defaultdict(lambda: defaultdict(list))
    for r in q:
        ep, tp = truth[r["query_id"]]
        if ep not in episodes:
            continue
        s = seq[r["machine_id"]]
        p = G.win_probability(r, s, gm.ST, gm._sp(s), cellprior=gm._cp(r), **gm._mcfg(s, r_type=r["machine_type"]))
        c = cond[r["query_id"]]; e[c][ep].append((p - tp) ** 2); b[c][ep].append((ref[r["query_id"]] - tp) ** 2)
    out = {}
    for c in CONDITIONS:
        if not e[c]:
            continue
        ee = sum(sum(v) / len(v) for v in e[c].values()) / len(e[c]); bb = sum(sum(v) / len(v) for v in b[c].values()) / len(b[c])
        out[c] = ee / max(bb, 1e-9)
    return out


def select(pub, out, guard_on=True):
    t0 = time.time(); tune, audit = folds(pub)
    gm0 = Model(cache=False)._fit_gate(pub); t1 = time.time(); print(f"[select] v8 gate fit {t1 - t0:.0f}s; structures kept: {[f'{f}{l}' for f, l in gm0.ST]}")
    gm1 = copy.deepcopy(gm0); info = attach_shared_base(gm1, *pooled_split(pub, ("train",))); t2 = time.time()
    print(f"[select] shared-base structure fit {t2 - t1:.0f}s; derived pairs (pair, size lag): {info.get('pairs')}")
    for t, d in info.get("per_type", {}).items():
        print(f"   {t:9s} gamma {d['gam']:.2f} slopes {d['b_on']:+.2f}/{d['b_off']:+.2f} sigma {d['ls_on']:.2f}/{d['ls_off']:.2f} mu {d['mu_on']:+.2f}/{d['mu_off']:+.2f} "
              f"converged={d['converged']} at_bound={d['at_bound']} start_spread={d['start_spread']}")
    print("   structure prior:", info.get("structure_prior"))
    r0, r1 = win_ratios(pub, gm0, tune), win_ratios(pub, gm1, tune)
    m0, m1 = sum(r0.values()) / len(r0), sum(r1.values()) / len(r1)
    worst = max(r1[c] / r0[c] - 1 for c in r0)
    print("[select] TUNING half, win-side ratio vs frozen reference (lower is better):")
    for c in r0:
        print(f"   {c:14s} base {r0[c]:.4f}  sb {r1[c]:.4f}  ({r1[c] / r0[c] - 1:+.1%})")
    print(f"   mean           base {m0:.4f}  sb {m1:.4f}  ({m1 / m0 - 1:+.1%});  worst condition {worst:+.1%}")
    choose = (m1 < m0 * (1 - MARGIN)) and (worst <= WORSE) and bool(info.get("pairs"))
    guard = None
    if choose and guard_on:                              # frozen choice: ONE audit look as a guard
        a0, a1 = win_ratios(pub, gm0, audit), win_ratios(pub, gm1, audit); am0, am1 = sum(a0.values()) / len(a0), sum(a1.values()) / len(a1)
        guard = dict(base=am0, sb=am1); print(f"[select] AUDIT guard (single look): base {am0:.4f}  sb {am1:.4f}")
        if am1 > am0:
            choose = False; print("   audit guard FAILED: sb is worse than base on audit -> falling back to plain v8")
    kw = {"shared_base": True} if choose else {}
    res = dict(chosen="sb" if choose else "base", kwargs=kw, tuning=dict(base=r0, sb=r1), audit_guard=guard, shared_fit={k: v for k, v in info.items()},
               seconds=round(time.time() - t0, 1))
    os.makedirs(out, exist_ok=True); json.dump(res, open(os.path.join(out, "chosen.json"), "w"), indent=1, default=float)
    print(f"[select] CHOSEN: {res['chosen']}  kwargs={kw}  ({res['seconds']}s)  -> {os.path.join(out, 'chosen.json')}")
    return res


def final(pub, out, kwargs, data_arg, tuning_only=False, workers=None):
    """Full v8 pipeline with `kwargs`. tuning_only: score the TUNING half only (used while comparing candidates; audit untouched)."""
    import pickle
    t0 = time.time(); kw = dict(kwargs, **({"workers": workers} if workers else {}))
    m = Model(cache=False, **kw).fit(pub); pred = m.predict(pub, "validation"); t1 = time.time()
    os.makedirs(out, exist_ok=True); pickle.dump(pred, open(os.path.join(out, "validation_pred.pkl"), "wb"))
    meta = json.load(open(os.path.join(pub, "metadata.json"))); maxp = float(meta.get("max_points", 60)); tune, audit = folds(pub)
    tp, tr = points(pub, pred, tune, maxp); res = dict(kwargs=kwargs, tuning=tp, tuning_ratios=tr)
    print(f"[final {kwargs}] fit + validation predict {t1 - t0:.0f}s; official TUNING {tp:.4f}  " + "  ".join(f"{c} {tr[c]:.5f}" for c in CONDITIONS), flush=True)
    if not tuning_only:
        ap, _ = points(pub, pred, audit, maxp); fp, fr = points(pub, pred, None, maxp); res.update(audit=ap, full=fp, ratios=fr)
        print(f"        audit {ap:.4f}  full {fp:.4f}  full ratios " + "  ".join(f"{c} {fr[c]:.5f}" for c in CONDITIONS))
    csv_path = os.path.join(out, "submission.csv"); write_predictions(csv_path, m.predict(pub, "test"))
    m.cache = False; m.__dict__.pop("_sz", None)               # checkpoint for method4_ml.make_entry --model-pkl (package without refitting)
    with open(os.path.join(out, "model.pkl"), "wb") as fh:
        pickle.dump(m, fh, protocol=pickle.HIGHEST_PROTOCOL)
    chk = subprocess.run([sys.executable, os.path.join(HERE, "..", "kit", "check_submission.py"), csv_path, "--data", pub], capture_output=True, text=True)
    res["checker_ok"] = chk.returncode == 0                      # a checker problem is REPORTED, never allowed to throw away a finished fit
    if not res["checker_ok"]:
        print(f"[final {kwargs}] WARNING kit checker failed: {(chk.stdout + chk.stderr).strip().splitlines()[-1:]}", flush=True)
    res["seconds"] = round(time.time() - t0, 1); print(f"[final {kwargs}] test CSV written (checker ok={res['checker_ok']}); {res['seconds']}s -> {csv_path}", flush=True)
    json.dump(res, open(os.path.join(out, "final.json"), "w"), indent=1, default=float)
    return res


FULL_MARGIN = 0.05   # official tuning points ON A 60-POINT SCALE the candidate must gain over plain v8 (scaled by max_points/60; halves differ by 0.02-0.19)


def full_select(pub, out, data_arg, guard_on=True, workers=4):
    """Run plain v8 and v8 + shared-base as two FULL pipelines in parallel; choose on the official TUNING metric; then ONE audit look for the choice."""
    import pickle, shutil
    t0 = time.time(); procs = {}
    for name, flag in (("base", "base"), ("sb", "sb")):
        cmd = [sys.executable, os.path.join(HERE, "adapt.py"), "--data", data_arg, "--out", os.path.join(out, name), "--stage", "final",
               "--force", flag, "--tuning-only-final", "--workers", str(workers)]
        procs[name] = subprocess.Popen(cmd, cwd=HERE)
    for name, p in procs.items():
        if p.wait() != 0:
            print(f"[full-select] {name} FAILED (exit {p.returncode})")
    R = {n: json.load(open(os.path.join(out, n, "final.json"))) for n in procs if os.path.exists(os.path.join(out, n, "final.json"))}
    if "base" not in R:
        sys.exit("plain v8 failed: no fallback produced")
    choose = "base"
    if "sb" in R:
        gain = R["sb"]["tuning"] - R["base"]["tuning"]
        worst = max(R["sb"]["tuning_ratios"][c] / R["base"]["tuning_ratios"][c] - 1 for c in CONDITIONS)
        print(f"[full-select] TUNING official: base {R['base']['tuning']:.4f}  sb {R['sb']['tuning']:.4f}  gain {gain:+.4f}  worst condition {worst:+.1%}")
        mp = float(json.load(open(os.path.join(pub, 'metadata.json'))).get('max_points', 60)); margin = FULL_MARGIN * mp / 60
        print(f'[full-select] margin {margin:.4f} points (max_points {mp:g})')
        if gain >= margin and worst <= WORSE:
            choose = "sb"
    meta = json.load(open(os.path.join(pub, "metadata.json"))); maxp = float(meta.get("max_points", 60)); tune, audit = folds(pub)
    if guard_on:                                                    # ONE audit look, after the choice is frozen
        pc = pickle.load(open(os.path.join(out, choose, "validation_pred.pkl"), "rb")); ap, _ = points(pub, pc, audit, maxp); fp, fr = points(pub, pc, None, maxp)
        print(f"[full-select] chosen {choose}: audit {ap:.4f}  full {fp:.4f}")
    shutil.copy(os.path.join(out, choose, "submission.csv"), os.path.join(out, "submission.csv"))
    json.dump(dict(chosen=choose, kwargs=R[choose]["kwargs"], tuning={n: R[n]["tuning"] for n in R}, seconds=round(time.time() - t0, 1)),
              open(os.path.join(out, "chosen.json"), "w"), indent=1)
    print(f"[full-select] CHOSEN {choose} ({time.time() - t0:.0f}s) -> {os.path.join(out, 'submission.csv')}; package with kwargs {R[choose]['kwargs']}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("all", "select", "final", "full-select"), default="all")
    ap.add_argument("--tuning-only-final", action="store_true"); ap.add_argument("--workers", type=int); ap.add_argument("--force", choices=("base", "sb"))
    ap.add_argument("--no-guard", action="store_true", help="skip the single audit-half guard (rehearsals on a kit whose audit half must stay untouched)")
    a = ap.parse_args(); pub = public_dir(a.data); meta, have = kit_check(pub)
    if a.stage == "full-select":
        return full_select(pub, a.out, a.data, guard_on=not a.no_guard, workers=a.workers or 4)
    if a.force:
        res = dict(chosen=a.force, kwargs={"shared_base": True} if a.force == "sb" else {}); os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, "chosen.json"), "w"))
    elif a.stage in ("all", "select"):
        if not have:
            print("no validation answers: cannot select; defaulting to plain v8"); res = dict(chosen="base", kwargs={}); os.makedirs(a.out, exist_ok=True)
            json.dump(res, open(os.path.join(a.out, "chosen.json"), "w"))
        else:
            res = select(pub, a.out, guard_on=not a.no_guard)
    else:
        res = json.load(open(os.path.join(a.out, "chosen.json")))
    if a.stage in ("all", "final"):
        final(pub, a.out, res["kwargs"], a.data, tuning_only=a.tuning_only_final, workers=a.workers)


if __name__ == "__main__":
    main()
