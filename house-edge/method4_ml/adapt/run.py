"""One-command round-two pipeline for method4_ml v9 (Person D).

  python3 -m method4_ml.adapt.run --data data/adaptation --out build/adapt

1. select: method4_ml.round2 fits v8, v9 (polarity bank) and v9 + polarity logistic term on the new kit, scores every
   variant on the new validation TUNING half, and keeps the default (v9 trees + logistic) unless another variant beats it by
   >= 0.05 tuning points with the audit half not worse. Writes <out>/select/{submission.csv, chosen.json, learned.json}.
2. entry: method4_ml.make_entry builds the submission ZIP with the chosen settings (refits on the runner, cache off,
   2 workers), checks it, reruns the unpacked ZIP and records hashes in <out>/entry/verification.json.
3. compares the entry CSV with the selection CSV (they must agree within 1e-8).
Add --wide to also try the widened gate (slower; check 2-core runtime), --quick for a fast rehearsal (bags 2, seeds 2).
"""
import argparse, csv, json, os, subprocess, sys


def read(path):
    with open(path, newline="") as f:
        return {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--wide", action="store_true"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-entry", action="store_true", help="selection only (no ZIP)")
    a = ap.parse_args(); out = os.path.abspath(a.out)
    if os.path.exists(out):
        sys.exit(f"{out} exists; choose a new --out folder")
    sel = os.path.join(out, "select"); os.makedirs(out)
    cmd = [sys.executable, "-m", "method4_ml.round2", "--data", a.data, "--out", sel, "--no-cache"]
    if a.wide: cmd.append("--wide")
    if a.quick: cmd += ["--set", json.dumps({"bags": 2, "seeds": 2})]
    print("+", " ".join(cmd), flush=True); subprocess.run(cmd, check=True)
    chosen = json.load(open(os.path.join(sel, "chosen.json")))
    print("chosen settings:", json.dumps(chosen), flush=True)
    if a.skip_entry:
        return
    ent = os.path.join(out, "entry")
    cmd = [sys.executable, "-m", "method4_ml.make_entry", "--data", a.data, "--out", ent, "--kwargs", json.dumps({**chosen, "workers": 2})]
    print("+", " ".join(cmd), flush=True); subprocess.run(cmd, check=True)
    csvs = [os.path.join(ent, f) for f in os.listdir(ent) if f.endswith(".csv")]
    if csvs:
        A, B = read(os.path.join(sel, "submission.csv")), read(csvs[0])
        diff = max(abs(x - y) for k in A for x, y in zip(A[k], B[k])) if A.keys() == B.keys() else float("inf")
        print(f"selection CSV vs entry CSV: max |diff| {diff:.2e} -> {'OK' if diff <= 1e-8 else 'MISMATCH: upload the entry/ pair only'}")
    print(f"upload: the CSV and ZIP inside {ent} (one build = one pair)")


if __name__ == "__main__":
    main()
