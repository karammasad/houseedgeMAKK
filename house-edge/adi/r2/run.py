"""Round-two runbook (adi). One command from the repo's house-edge/ folder:

  python -m adi.r2.run --data data/adaptation --out build/r2

1. Checks the kit (metadata round, splits present).
2. Fits the candidates on the new kit's train data:
     r2    adi.r2      wide, re-learned structure family (see adi/r2/model.py)
     main  adi.sizewin the exact 58.21 main-round model settings (safety net)
     cook  adi.cook    adi.r2 + two-level hierarchical size-split cells on short logs (kappa per type by train
                       evidence) + exact two-vector size menu learned from history ticks (port of method4 v8)
                       + GateSizes' size-lag coupling driven by adi's win posterior (adi round 13)
                       (main data: tuning 58.318 / audit 58.279 / full 58.297)
     cook_nomenu       the same without the size menu, in case round two's sizes stop being two-state
                       (main data: 58.277 / 58.208 / 58.243)
3. Scores each on the new validation set (tuning / audit / full, per condition) with the official metric.
4. Picks the candidate with the higher TUNING-half score (the audit half stays an honest check; a candidate is only
   preferred if its audit doesn't drop by more than 0.3 points relative to the best other audit).
5. Prints what the chosen model learned (structures, cell priors, signal slopes, flip prior, correction).
6. Writes <out>/submission.csv (test), runs the kit checker, and builds the verified upload ZIP with
   adi.make_entry (fresh-unzip rerun must be byte-identical) into <out>/entry/.
Upload <out>/entry/submission.csv and <out>/entry/entry.zip (identical CSV to <out>/submission.csv).

Options: --only r2|main|cook|cook_nomenu (skip the comparison), --no-build (skip ZIP), --workers N.
"""
import argparse, importlib, json, os, subprocess, sys, time
from common.data import public_dir, write_predictions
from common.evaluate import folds, points, CONDITIONS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDIDATES = {"r2": ("adi.r2", {}), "main": ("adi.sizewin", {}), "cook": ("adi.cook", {"inner": "r2", "size_couple": "adi"}),
              "cook_nomenu": ("adi.cook", {"inner": "r2", "size_couple": "adi", "size_menu": False})}


def describe(m):
    w = m.base.win; out = []
    top = sorted(zip(list(w.ST) + ["none"], w.sprior), key=lambda x: -x[1])[:8]
    out.append("  structure prior (top): " + ", ".join(f"{s}: {p:.3f}" for s, p in top))
    sp = getattr(w, "signal_priors", None) or {}
    for t, v in sorted(sp.items()):
        out.append(f"  cell prior {t:8s} on {v[0]:.2f} (k {v[2]:g}, slope {v[4]:+.2f})  off {v[1]:.3f} (k {v[3]:g}, slope {v[5]:+.2f})")
    out.append(f"  short-log flip prior p_change_short = {w.p_change_short}")
    out.append("  size-state correction coef = " + ", ".join(f"{c:+.3f}" for c in m.coef))
    if getattr(m, "kappa_fit", None) is not None:
        out.append(f"  cook short-log sub-cell kappa per type = {m.kappa_fit}  (train evidence vs off: {getattr(m, 'kappa_evidence', '')})")
    if getattr(m, "menu", None):
        for t, (v1, v2) in sorted(m.menu.items()):
            out.append(f"  size menu {t:8s} V1 " + " ".join(f"{x:.3f}" for x in v1) + "   V2 " + " ".join(f"{x:.3f}" for x in v2))
    tw = getattr(m.base.sizes, "tie_weight", None)
    if tw is not None:
        out.append(f"  GateSizes tie weight = {tw}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=sorted(CANDIDATES)); ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    pub = public_dir(a.data)
    meta = json.load(open(os.path.join(pub, "metadata.json")))
    print(f"kit: round={meta.get('round')} season={meta.get('season_id')} max_points={meta.get('max_points')}")
    if meta.get("round") != "adaptation":
        print("WARNING: metadata round is not 'adaptation' -- is this the right kit?")
    for sp in ("train", "validation", "test"):
        if not os.path.exists(os.path.join(pub, sp, "queries.csv")):
            sys.exit(f"missing {sp}/queries.csv in {pub}")
    os.makedirs(a.out, exist_ok=False)
    maxp = float(meta.get("max_points", 40)); has_t = os.path.exists(os.path.join(pub, "validation", "targets.csv"))
    tune, audit = folds(pub) if has_t else (None, None)
    res = {}
    for name in ([a.only] if a.only else ["r2", "main", "cook", "cook_nomenu"]):
        method, kw = CANDIDATES[name]; kw = dict(kw)
        if a.workers:
            kw["workers"] = a.workers
        t0 = time.time()
        m = importlib.import_module(method + ".model").Model(**kw).fit(pub)
        pv = m.predict(pub, "validation"); secs = time.time() - t0
        line = f"[{name}] {method}: fit+validation {secs:.0f}s"
        if has_t:
            tp, _ = points(pub, pv, tune, maxp); ap_, _ = points(pub, pv, audit, maxp); fp, fr = points(pub, pv, None, maxp)
            res[name] = (tp, ap_, m, method, kw)
            line += f"  tuning {tp:.3f}  audit {ap_:.3f}  full {fp:.3f}  " + " ".join(f"{c} {fr[c]:.4f}" for c in CONDITIONS)
        else:
            res[name] = (0.0, 0.0, m, method, kw)
        print(line, flush=True)
    names = list(res)
    order = sorted(names, key=lambda n: -res[n][0])
    best = order[0]
    for n in order:  # best tuning whose audit is within 0.3 of the best audit among the others
        others = [res[o][1] for o in names if o != n]
        if not others or res[n][1] >= max(others) - 0.3:
            best = n; break
        print(f"tuning prefers {n} but its audit is >0.3 below another candidate's; skipping it")
    tp, ap_, m, method, kw = res[best]
    print(f"CHOSEN: {best} ({method})\n{describe(m)}")
    csv_path = os.path.join(a.out, "submission.csv")
    write_predictions(csv_path, m.predict(pub, "test"))
    subprocess.run([sys.executable, os.path.join(HERE, "kit", "check_submission.py"), csv_path, "--data", pub], check=True)
    with open(os.path.join(a.out, "choice.json"), "w") as f:
        json.dump({"chosen": best, "method": method, "kwargs": kw,
                   "scores": {n: {"tuning": r[0], "audit": r[1]} for n, r in res.items()}}, f, indent=2)
    if not a.no_build:
        kw_build = {k: v for k, v in kw.items() if k != "workers"}
        subprocess.run([sys.executable, "-m", "adi.make_entry", "--method", method, "--kwargs", json.dumps(kw_build),
                        "--data", a.data, "--out", os.path.join(a.out, "entry")], check=True, cwd=HERE)
        print(f"UPLOAD: {os.path.join(a.out, 'entry', 'submission.csv')} and {os.path.join(a.out, 'entry', 'entry.zip')}")


if __name__ == "__main__":
    main()
