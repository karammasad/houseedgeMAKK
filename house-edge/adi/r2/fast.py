"""Fast round-two runbook (adi): all candidates fit IN PARALLEL, then one verified build. From house-edge/:

  python3 -m adi.r2.fast --data data/adaptation --out build/r2fast

1. Checks the kit (round, splits).
2. Starts one process per candidate (adi.r2.cand), each with cores / n_candidates workers; every process fits on the new
   train data and writes validation + test predictions (separate processes: adi.cook's in-process patches stay isolated).
3. Scores each on the new validation set (tuning / audit / full, per condition), official metric.
4. Picks the best TUNING score whose audit is within 0.3 of the best other audit (as adi.r2.run).
5. Writes <out>/submission.csv from the chosen candidate's own fit, runs the kit checker, and builds the ZIP with
   adi.make_entry --csv: the ZIP is unpacked into a fresh folder and rerun once; it must reproduce the CSV byte for byte.
Upload <out>/entry/submission.csv and <out>/entry/entry.zip.

Options: --only NAME[,NAME] (subset), --no-build, --cores N (default: all).
"""
import argparse, json, os, subprocess, sys, time
from common.data import public_dir, write_predictions
from common.evaluate import folds, points, CONDITIONS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDIDATES = {
    "r2": ("adi.r2", {}),
    "cook": ("adi.cook", {"inner": "r2", "size_couple": "adi"}),
    "cook_nomenu": ("adi.cook", {"inner": "r2", "size_couple": "adi", "size_menu": False}),
    "cook_nocouple": ("adi.cook", {"inner": "r2"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only"); ap.add_argument("--no-build", action="store_true"); ap.add_argument("--cores", type=int)
    a = ap.parse_args(); t_start = time.time()
    pub = public_dir(a.data); meta = json.load(open(os.path.join(pub, "metadata.json")))
    print(f"kit: round={meta.get('round')} season={meta.get('season_id')} max_points={meta.get('max_points')}", flush=True)
    if meta.get("round") != "adaptation":
        print("WARNING: metadata round is not 'adaptation' -- is this the right kit?")
    for sp in ("train", "validation", "test"):
        if not os.path.exists(os.path.join(pub, sp, "queries.csv")):
            sys.exit(f"missing {sp}/queries.csv in {pub}")
    os.makedirs(a.out, exist_ok=False)
    names = a.only.split(",") if a.only else list(CANDIDATES)
    cores = a.cores or os.cpu_count() or 2; per = max(2, cores // len(names))
    procs = {}
    for n in names:
        method, kw = CANDIDATES[n]; kw = dict(kw, workers=per); d = os.path.join(a.out, "cand_" + n)
        os.makedirs(d); log = open(os.path.join(d, "log.txt"), "w")
        procs[n] = (subprocess.Popen([sys.executable, "-m", "adi.r2.cand", "--method", method, "--kwargs", json.dumps(kw),
                                      "--data", a.data, "--out", d], cwd=HERE, stdout=log, stderr=subprocess.STDOUT), log)
    print(f"fitting {len(names)} candidates in parallel ({per} workers each): {', '.join(names)}", flush=True)
    for n, (p, log) in procs.items():
        p.wait(); log.close()
        print(f"  {n}: exit {p.returncode}  ({time.time() - t_start:.0f}s)", flush=True)
    maxp = float(meta.get("max_points", 40)); has_t = os.path.exists(os.path.join(pub, "validation", "targets.csv"))
    tune, audit = folds(pub) if has_t else (None, None); res = {}
    for n in names:
        f = os.path.join(a.out, "cand_" + n, "pred.json")
        if procs[n][0].returncode != 0 or not os.path.exists(f):
            print(f"[{n}] FAILED, see {os.path.join(a.out, 'cand_' + n, 'log.txt')}"); continue
        P = json.load(open(f))
        if has_t:
            tp = points(pub, P["validation"], tune, maxp)[0]; au = points(pub, P["validation"], audit, maxp)[0]
            fp, fr = points(pub, P["validation"], None, maxp)
            print(f"[{n}] tuning {tp:.3f}  audit {au:.3f}  full {fp:.3f}  " + " ".join(f"{c} {fr[c]:.4f}" for c in CONDITIONS)
                  + f"  ({P['seconds']:.0f}s)")
        else:
            tp = au = 0.0
        res[n] = (tp, au, P)
    if not res:
        sys.exit("every candidate failed")
    order = sorted(res, key=lambda n: -res[n][0]); best = order[0]
    for n in order:
        others = [res[o][1] for o in res if o != n]
        if not others or res[n][1] >= max(others) - 0.3:
            best = n; break
        print(f"tuning prefers {n} but its audit is >0.3 below another candidate's; skipping it")
    method, kw = CANDIDATES[best]
    print(f"CHOSEN: {best} ({method} {json.dumps(kw)})")
    print(open(os.path.join(a.out, "cand_" + best, "learned.txt")).read())
    csv_path = os.path.join(a.out, "submission.csv")
    write_predictions(csv_path, res[best][2]["test"])
    subprocess.run([sys.executable, os.path.join(HERE, "kit", "check_submission.py"), csv_path, "--data", pub], check=True)
    with open(os.path.join(a.out, "choice.json"), "w") as f:
        json.dump({"chosen": best, "method": method, "kwargs": kw,
                   "scores": {n: {"tuning": r[0], "audit": r[1]} for n, r in res.items()}}, f, indent=2)
    print(f"candidates + choice: {time.time() - t_start:.0f}s", flush=True)
    if not a.no_build:
        subprocess.run([sys.executable, "-m", "adi.make_entry", "--method", method, "--kwargs", json.dumps(kw),
                        "--csv", os.path.abspath(csv_path), "--data", a.data, "--out", os.path.join(a.out, "entry")],
                       check=True, cwd=HERE)
        print(f"UPLOAD: {os.path.join(a.out, 'entry', 'submission.csv')} and {os.path.join(a.out, 'entry', 'entry.zip')}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
