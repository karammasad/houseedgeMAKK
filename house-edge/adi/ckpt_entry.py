"""Checkpoint entry builder (adi): fit ONCE here, ship the fitted model, and the organizers' rerun only loads + predicts.

The participant guide endorses this ("Saving a trained checkpoint is usually easier than retraining during the check").
It moves the expensive fit (method4's feature building alone is ~18 min on 2 cores) out of the 30-minute / 2-core rerun.

  build/venv312/bin/python -m adi.ckpt_entry --method adi.pool --data data/adaptation --out build/pool_entry

Steps: stage the source (common/, adi/ incl. subpackages, method*/) -> fit Model(**kwargs) on <data>/train (+ histories)
-> pickle it to checkpoint.pkl -> write predict.py (loads the checkpoint, 2 workers, 2 BLAS/OpenMP threads),
reproduce.json, entry_config.json, requirements.txt (EXACT installed versions of the scikit-learn stack) and README.md
-> predict test in a fresh process from the staged folder -> kit checker -> zip with kit/build_entry.py -> unpack into a
fresh folder, rerun, require a byte-identical CSV. Optionally --val-out writes the fitted model's validation predictions
(JSON) for scoring. Upload <out>/submission.csv and <out>/entry.zip.
"""
import argparse, filecmp, hashlib, importlib, json, os, pickle, shutil, subprocess, sys, tempfile, time, zipfile
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
PREDICT = '''"""Organizer entry point: python predict.py --data /data --out /output/submission.csv
Loads the fitted checkpoint (no retraining) and predicts the test split. Parallelism is capped at 2 (results do not depend on it)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import argparse, json, pickle, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    import numpy  # noqa: F401  (the checkpoint's scikit-learn stack, installed from requirements.txt)
except ImportError:
    pass
from common.data import public_dir, write_predictions
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(os.path.join(HERE, "checkpoint.pkl"), "rb") as f:
    model = pickle.load(f)
if hasattr(model, "set_workers"):
    model.set_workers(2)
elif hasattr(model, "workers"):
    model.workers = 2
write_predictions(a.out, model.predict(public_dir(a.data), "test"))
'''


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)


def stage(src):
    for pkg in ["common", "adi"] + sorted(d for d in os.listdir(ROOT) if d.startswith("method") and os.path.isdir(os.path.join(ROOT, d))):
        base = os.path.join(ROOT, pkg)
        for dirpath, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if os.path.isfile(os.path.join(dirpath, d, "__init__.py")))
            rel = os.path.relpath(dirpath, ROOT); os.makedirs(os.path.join(src, rel), exist_ok=True)
            for f in files:
                if f.endswith(".py") or (f.endswith(".json") and not f.endswith(".sk_check.json")):
                    shutil.copy(os.path.join(dirpath, f), os.path.join(src, rel, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--val-out")
    ap.add_argument("--load", help="an already-fitted, pickled model to ship (skips the fit; --method/--kwargs are recorded only)")
    a = ap.parse_args(); t0 = time.time()
    out = os.path.abspath(a.out); src = os.path.join(out, "entry_src")
    if os.path.exists(out):
        sys.exit(f"{out} exists; choose a new --out folder")
    os.makedirs(src); stage(src)
    sys.path.insert(0, ROOT)
    from common.data import public_dir
    pub = public_dir(os.path.abspath(a.data)); kw = json.loads(a.kwargs)
    if a.load:
        importlib.import_module(a.method + ".model")
        with open(a.load, "rb") as f:
            model = pickle.load(f)
        print(f"loaded fitted model {a.load}", flush=True)
    else:
        model = importlib.import_module(a.method + ".model").Model(**kw).fit(pub)
        print(f"fit {time.time() - t0:.0f}s", flush=True)
    if a.val_out:
        with open(a.val_out, "w") as f:
            json.dump({"validation": model.predict(pub, "validation")}, f)
    with open(os.path.join(src, "checkpoint.pkl"), "wb") as f:
        pickle.dump(model, f, protocol=4)
    pins = []
    needs = any(m in sys.modules for m in ("sklearn", "numpy", "scipy"))   # stdlib-only models ship no requirements
    for p in (PINS if needs else ()):
        try:
            pins.append(f"{p}=={metadata.version(p)}")
        except metadata.PackageNotFoundError:
            pass
    if pins:
        with open(os.path.join(src, "requirements.txt"), "w") as f:
            f.write("\n".join(pins) + "\n")
    with open(os.path.join(src, "predict.py"), "w") as f:
        f.write(PREDICT)
    with open(os.path.join(src, "entry_config.json"), "w") as f:
        json.dump({"method": a.method, "kwargs": kw, "checkpoint": "checkpoint.pkl"}, f, indent=2)
    with open(os.path.join(src, "reproduce.json"), "w") as f:
        json.dump({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, f, indent=2)
    csv_path = os.path.join(out, "submission.csv"); t1 = time.time()
    run([sys.executable, "predict.py", "--data", pub, "--out", csv_path], cwd=src)
    t_pred = time.time() - t1
    digest = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    ck = os.path.getsize(os.path.join(src, "checkpoint.pkl")) / 1e6
    with open(os.path.join(src, "README.md"), "w") as f:
        f.write(f"""# House Edge entry (saved checkpoint)

Recreate the forecast (run from this folder):

    python predict.py --data /data --out /output/submission.csv

- Expected output SHA-256: `{digest}`
- Model: `{a.method}` with settings `{json.dumps(kw)}` (entry_config.json), fitted on `<data>/train` (labels + histories) and the
  HISTORIES of validation and test (no validation answers), then saved as `checkpoint.pkl` ({ck:.1f} MB, pickle protocol 4).
  `predict.py` loads the checkpoint and predicts `<data>/test`; it does not retrain.
- Dependencies (`requirements.txt`, exact versions used to fit and pickle the checkpoint): {", ".join(pins) or "standard library only"}.
  Built with Python {sys.version.split()[0]}; the runner's Python 3.13 loads the same pickle.
- Parallelism: 2 workers, 2 BLAS/OpenMP threads (set in predict.py); results do not depend on it. Local rerun time: {t_pred:.0f} s.
- Seeds: every random choice uses fixed seeds (string-seeded `random.Random`, fixed `random_state`); reruns are byte-identical.
- Writes only the `--out` file; reads only `<data>/test` queries and histories at prediction time.
""")
    run([sys.executable, os.path.join(ROOT, "kit", "check_submission.py"), csv_path, "--data", pub])
    zip_path = os.path.join(out, "entry.zip")
    run([sys.executable, os.path.join(ROOT, "kit", "build_entry.py"), "--source", src, "--out", zip_path], cwd=os.path.join(ROOT, "kit"))
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp); again = os.path.join(tmp, "rerun.csv")
        run([sys.executable, "predict.py", "--data", pub, "--out", again], cwd=tmp)
        if not filecmp.cmp(again, csv_path, shallow=False):
            sys.exit("RERUN MISMATCH: the ZIP does not reproduce the CSV")
    print(f"\nOK. Upload {csv_path} and {zip_path}. CSV SHA-256 {digest}. Checkpoint {ck:.1f} MB, "
          f"predict-only rerun {t_pred:.0f} s, total build {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
