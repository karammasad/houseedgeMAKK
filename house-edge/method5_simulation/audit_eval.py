"""THE ONE audit run for the selected frozen Method 5 candidate (final confirmation; no tuning afterwards).

  python -m method5_simulation.audit_eval --pred method5_simulation/out/validation_s2.csv --name "S2"

Scores the candidate's validation predictions (already written by the frozen pipeline, trained on train only) with the
exact competition formula on the tuning half, the AUDIT half and the FULL validation set, and with the organizers'
kit checker. Refuses to run twice for the same candidate (writes out/audit_done_<name>.txt).
"""
import argparse, csv, os, subprocess, sys
from common.data import public_dir
from common.evaluate import folds, points, COLS, CONDITIONS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--name", required=True); ap.add_argument("--data", default="data/main")
    a = ap.parse_args(); pub = public_dir(a.data); lock = f"method5_simulation/out/audit_done_{a.name}.txt"
    if os.path.exists(lock):
        sys.exit(f"audit already run for {a.name}: {open(lock).read().strip()}")
    P = {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(a.pred))}
    tune, audit = folds(pub); tp, tr = points(pub, P, tune); ap_, ar = points(pub, P, audit); fp, fr = points(pub, P)
    line = f"{a.name}: tuning {tp:.3f} | AUDIT {ap_:.3f} | full validation {fp:.3f}"
    print(line); print("  full-validation ratios: " + "  ".join(f"{c[:6]} {fr[c]:.4f}" for c in CONDITIONS))
    print("  audit ratios:           " + "  ".join(f"{c[:6]} {ar[c]:.4f}" for c in CONDITIONS)); open(lock, "w").write(line)
    subprocess.run([sys.executable, "kit/check_submission.py", a.pred, "--data", pub, "--split", "validation"], check=False)


if __name__ == "__main__":
    main()
