"""Control panel for this folder: score, keep-check, promote, table. Run from mominsreverseengineering/.

  python3 run.py score  --name "baseline v8"  [--method method4_ml] [--kwargs '{"blend":0.5}'] [--data ../data/main] [--test-out PATH]
  python3 run.py keep   --b '{"method":"method4_ml","kwargs":{...}}'   # vs FINAL in final.json (uses method2_sizes.keepcheck)
  python3 run.py table                                                  # standard results table from experiments/ledger.csv
  python3 run.py final  --set '{"method":"method4_ml","kwargs":{...}}' --why "what was kept, and the FINDINGS heading"

Every score appends one row to experiments/ledger.csv. Nothing here writes outside this folder, except --test-out.
Validation targets are read only by `score` (for points) and `keep`; fit and predict never see them.
"""
import argparse, csv, datetime, importlib, json, os, subprocess, sys, time
from common.data import public_dir, write_predictions
from common.evaluate import folds, points, CONDITIONS

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "experiments", "ledger.csv")
FINAL = os.path.join(HERE, "final.json")
LEDGER_COLS = ["time", "name", "method", "kwargs", "tuning", "audit", "full", "stable", "short", "drift", "shared", "seconds"]


def _load_final():
    with open(FINAL) as f:
        return json.load(f)


def _append(row):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_score(a):
    pub = public_dir(a.data)
    t0 = time.time()
    m = importlib.import_module(a.method + ".model").Model(**json.loads(a.kwargs)).fit(pub)
    pred = m.predict(pub, "validation"); secs = time.time() - t0
    tune, audit = folds(pub)
    tp = points(pub, pred, tune)[0]; ap = points(pub, pred, audit)[0]; fp, fr = points(pub, pred)
    row = dict(time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), name=a.name, method=a.method, kwargs=a.kwargs,
               tuning=round(tp, 4), audit=round(ap, 4), full=round(fp, 4),
               stable=round(fr["stable"], 5), short=round(fr["short_history"], 5), drift=round(fr["drift"], 5),
               shared=round(fr["shared_change"], 5), seconds=round(secs, 1))
    _append(row)
    print(f"{a.name}: tuning {tp:.3f}  audit {ap:.3f}  full {fp:.3f}  ({secs:.0f}s)  -> {os.path.relpath(LEDGER, HERE)}")
    if a.test_out:
        write_predictions(a.test_out, m.predict(pub, "test"))
        print(f"test predictions -> {a.test_out}")


def cmd_keep(a):
    fin = _load_final()
    cmd = [sys.executable, "-m", "method2_sizes.keepcheck", "--data", a.data,
           "--a", json.dumps(fin), "--b", a.b]
    subprocess.run(cmd, check=True, cwd=HERE)


def cmd_table(a):
    if not os.path.exists(LEDGER):
        sys.exit("no ledger yet: run `score` first")
    rows = list(csv.DictReader(open(LEDGER)))
    fin = _load_final()
    final_name = fin.get("name", "FINAL")
    print("**FINAL (what would be submitted)**\n")
    print("| run | tuning | audit | full | stable | short | drift | shared |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    cand = [r for r in rows if r["name"] == final_name]
    if cand:
        r = cand[-1]
        print(f"| {r['name']} | {r['tuning']} | {r['audit']} | {r['full']} | {r['stable']} | {r['short']} | {r['drift']} | {r['shared']} |")
    print("\n**History**\n")
    print("| run | tuning | audit | full | stable | short | drift | shared |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['time']} {r['name']} | {r['tuning']} | {r['audit']} | {r['full']} | {r['stable']} | {r['short']} | {r['drift']} | {r['shared']} |")


def cmd_final(a):
    new = json.loads(a.set)
    new["name"] = a.name
    new["why"] = a.why
    with open(FINAL, "w") as f:
        json.dump(new, f, indent=2)
    print(f"FINAL updated -> {os.path.relpath(FINAL, HERE)}. Add a TASK.md status line and a FINDINGS.md entry.")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--name", required=True); s.add_argument("--method", default="method4_ml")
    s.add_argument("--kwargs", default="{}"); s.add_argument("--data", default="../data/main"); s.add_argument("--test-out")
    s.set_defaults(fn=cmd_score)
    k = sub.add_parser("keep"); k.add_argument("--b", required=True); k.add_argument("--data", default="../data/main")
    k.set_defaults(fn=cmd_keep)
    t = sub.add_parser("table"); t.set_defaults(fn=cmd_table)
    f = sub.add_parser("final"); f.add_argument("--set", required=True); f.add_argument("--name", required=True)
    f.add_argument("--why", required=True); f.set_defaults(fn=cmd_final)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
