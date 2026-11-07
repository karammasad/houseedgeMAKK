"""Method 2's copy of tools/make_entry.py (Person B). Identical, except it ALSO packages adi/.

Why: tools/make_entry.py stages only common/ and method*/ folders, so any entry that uses Adi's gate
(e.g. method2_sizes with win_model="adi") fails on the runner with "No module named 'adi'".
The shared tools/make_entry.py is left untouched; this copy lives in method2_sizes/ only.

Usage (from the repo root):
  python -m method2_sizes.make_entry --method method2_sizes --kwargs "$(python3 -c 'import json,method2_sizes.round2 as r; print(json.dumps(r.FINAL))')" --data data/main --out build/main_m2

Steps: stage common/ and all method*/ packages plus a generated predict.py, README.md, reproduce.json and
entry_config.json; predict the test split; run the kit checker; zip with kit/build_entry.py; unpack the ZIP
into a fresh folder, rerun it, and require a byte-identical CSV. Upload <out>/submission.csv and <out>/entry.zip.
"""
import argparse, filecmp, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_PKGS = ["adi"]   # non-method* packages the recommended model imports
PREDICT = '''"""Organizer entry point: python predict.py --data /data --out /output/submission.csv"""
import argparse, importlib, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common.data import public_dir, write_predictions
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(os.path.join(HERE, "entry_config.json")) as f:
    cfg = json.load(f)
pub = public_dir(a.data)
model = importlib.import_module(cfg["method"] + ".model").Model(**cfg["kwargs"]).fit(pub)
write_predictions(a.out, model.predict(pub, "test"))
'''


def run(cmd, **kw):
    print("+", " ".join(cmd)); subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out); src = os.path.join(out, "entry_src")
    if os.path.exists(out):
        sys.exit(f"{out} exists; choose a new --out folder")
    os.makedirs(src)
    for pkg in ["common"] + [p for p in EXTRA_PKGS if os.path.isdir(os.path.join(ROOT, p))] + sorted(d for d in os.listdir(ROOT) if d.startswith("method") and os.path.isdir(os.path.join(ROOT, d))):
        os.makedirs(os.path.join(src, pkg))
        for f in os.listdir(os.path.join(ROOT, pkg)):
            if f.endswith(".py"):
                shutil.copy(os.path.join(ROOT, pkg, f), os.path.join(src, pkg, f))
    with open(os.path.join(src, "predict.py"), "w") as f:
        f.write(PREDICT)
    with open(os.path.join(src, "entry_config.json"), "w") as f:
        json.dump({"method": a.method, "kwargs": json.loads(a.kwargs)}, f, indent=2)
    with open(os.path.join(src, "reproduce.json"), "w") as f:
        json.dump({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, f, indent=2)
    csv_path = os.path.join(out, "submission.csv")
    run([sys.executable, "predict.py", "--data", os.path.abspath(a.data), "--out", csv_path], cwd=src)
    digest = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    with open(os.path.join(src, "README.md"), "w") as f:
        f.write(f"""# House Edge entry

- Method: `{a.method}` with settings `{a.kwargs}` (see entry_config.json)
- Environment: "python" image, Python 3.13, standard library only{" - except method4_ml, which needs scikit-learn" if a.method == "method4_ml" else ""}
- Forecast SHA-256: {digest}

Recreate: `python predict.py --data /data --out /output/submission.csv` (run from this folder).
It refits from `/data/train` (deterministic, no randomness) and predicts `/data/test`. No saved checkpoint is needed.
""")
    pub = os.path.join(os.path.abspath(a.data), "public") if os.path.isdir(os.path.join(a.data, "public")) else os.path.abspath(a.data)
    run([sys.executable, os.path.join(ROOT, "kit", "check_submission.py"), csv_path, "--data", pub])
    zip_path = os.path.join(out, "entry.zip")
    run([sys.executable, os.path.join(ROOT, "kit", "build_entry.py"), "--source", src, "--out", zip_path], cwd=os.path.join(ROOT, "kit"))
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp)
        again = os.path.join(tmp, "rerun.csv")
        run([sys.executable, "predict.py", "--data", os.path.abspath(a.data), "--out", again], cwd=tmp)
        if not filecmp.cmp(again, csv_path, shallow=False):
            sys.exit("RERUN MISMATCH: the ZIP does not reproduce the CSV")
    print(f"\nOK. Upload {csv_path} and {zip_path}. CSV SHA-256 {digest}")


if __name__ == "__main__":
    main()
