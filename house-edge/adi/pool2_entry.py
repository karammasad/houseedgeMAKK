"""Two-process pooled checkpoint entry (adi): average two ALREADY-FITTED models that cannot share one Python process.

Why: Momin's shared-base v8 (mominsreverseengineering/) ships its own copies of adi/ and method4_ml/ (its gate carries the 'xsg'
edits), while adi.cook patches adi.gate / GateSizes in-process. Same package names, overlapping patches: they must not be
imported together. Here each member lives in its own folder in the ZIP (its own code copy + its own checkpoint.pkl), and
predict.py runs each in a SEPARATE Python process (fresh interpreter, its own sys.path), then averages:
    P = w * A + (1 - w) * B   over all five probabilities (w fixed in advance; 0.5)

  build/venv312/bin/python -m adi.pool2_entry --a-root <momin root> --a-pkl <A model.pkl> --a-pred <A pred.json> \
      --b-root house-edge --b-pkl <B model.pkl> --b-pred <B pred.json> --data data/main --out build/pool2_entry

Checks: the CSV is predicted by predict.py from the staged folder (fresh processes), must equal w*A + (1-w)*B of the members'
own in-process test predictions (max diff 0), passes the kit checker, and a fresh unzip must reproduce it byte for byte.
"""
import argparse, filecmp, hashlib, json, os, shutil, subprocess, sys, tempfile, time, zipfile
from importlib import metadata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
CHILD = r'''
import json, os, pickle, sys
root, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root); sys.dont_write_bytecode = True
from common.data import public_dir
with open(os.path.join(root, "checkpoint.pkl"), "rb") as f:
    m = pickle.load(f)
if hasattr(m, "set_workers"):
    m.set_workers(2)
if hasattr(m, "workers"):
    m.workers = 2
inner = getattr(m, "inner", None)
if inner is not None and hasattr(inner, "workers"):
    inner.workers = 2
with open(out, "w") as f:
    json.dump(m.predict(public_dir(data), "test"), f)
'''
PREDICT = '''"""Organizer entry point: python predict.py --data /data --out /output/submission.csv
Two saved models, each run in its own Python process from its own folder (member_a/, member_b/); their test forecasts are
averaged: P = W * A + (1 - W) * B. No retraining. Scratch files go to the system temp folder (/tmp)."""
import argparse, json, os, subprocess, sys, tempfile
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
W = __W__
CHILD = __CHILD__
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    env[v] = "2"
preds = []
with tempfile.TemporaryDirectory() as tmp:
    for member in ("member_a", "member_b"):
        root = os.path.join(HERE, member); out = os.path.join(tmp, member + ".json")
        env_m = dict(env, PYTHONPATH=root)
        subprocess.run([sys.executable, "-c", CHILD, root, os.path.abspath(a.data), out], cwd=root, env=env_m, check=True)
        with open(out) as f:
            preds.append(json.load(f))
A, B = preds
P = {k: [W * x + (1 - W) * y for x, y in zip(A[k], B[k])] for k in A}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w", newline="") as f:
    f.write("query_id,p_0,p_1,p_2,p_5,p_20\\r\\n")
    for k in sorted(P):
        f.write(",".join([k] + [format(v, ".17g") for v in P[k]]) + "\\r\\n")
'''


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)


def stage(root, dst):
    for pkg in sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "__init__.py"))
                      and (d in ("common", "adi") or d.startswith("method"))):
        for dirpath, dirs, files in os.walk(os.path.join(root, pkg)):
            dirs[:] = sorted(d for d in dirs if os.path.isfile(os.path.join(dirpath, d, "__init__.py")))
            rel = os.path.relpath(dirpath, root); os.makedirs(os.path.join(dst, rel), exist_ok=True)
            for f in files:
                if f.endswith(".py") or (f.endswith(".json") and not f.endswith(".sk_check.json")):
                    shutil.copy(os.path.join(dirpath, f), os.path.join(dst, rel, f))


def main():
    ap = argparse.ArgumentParser()
    for m in ("a", "b"):
        ap.add_argument(f"--{m}-root", required=True); ap.add_argument(f"--{m}-pkl", required=True)
        ap.add_argument(f"--{m}-pred", required=True); ap.add_argument(f"--{m}-name", default=m)
    ap.add_argument("--w", type=float, default=0.5); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t0 = time.time()
    out = os.path.abspath(a.out); src = os.path.join(out, "entry_src")
    if os.path.exists(out):
        sys.exit(f"{out} exists; choose a new --out folder")
    os.makedirs(src)
    for m in ("a", "b"):
        d = os.path.join(src, f"member_{m}"); os.makedirs(d)
        stage(os.path.abspath(getattr(a, f"{m}_root")), d)
        shutil.copy(getattr(a, f"{m}_pkl"), os.path.join(d, "checkpoint.pkl"))
    with open(os.path.join(src, "predict.py"), "w") as f:
        f.write(PREDICT.replace("__W__", repr(a.w)).replace("__CHILD__", repr(CHILD)))
    with open(os.path.join(src, "reproduce.json"), "w") as f:
        json.dump({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, f, indent=2)
    pins = [f"{p}=={metadata.version(p)}" for p in PINS]
    with open(os.path.join(src, "requirements.txt"), "w") as f:
        f.write("\n".join(pins) + "\n")
    pub = os.path.abspath(a.data); pub = os.path.join(pub, "public") if os.path.isdir(os.path.join(pub, "public")) else pub
    csv_path = os.path.join(out, "submission.csv"); t1 = time.time()
    run([sys.executable, "predict.py", "--data", pub, "--out", csv_path], cwd=src); t_pred = time.time() - t1
    # must equal the pool of the members' own in-process test predictions
    A = json.load(open(a.a_pred))["test"]; B = json.load(open(a.b_pred))["test"]
    exp = {k: [a.w * x + (1 - a.w) * y for x, y in zip(A[k], B[k])] for k in A}
    import csv as _csv
    got = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in _csv.DictReader(open(csv_path))}
    dmax = max(abs(x - y) for k in exp for x, y in zip(exp[k], got[k]))
    print(f"CSV vs pool of in-process member predictions: max |diff| {dmax:.2e}")
    if dmax > 1e-12:
        sys.exit("CHECKPOINT MISMATCH vs in-process predictions")
    digest = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    sizes = {m: os.path.getsize(os.path.join(src, f"member_{m}", "checkpoint.pkl")) / 1e6 for m in ("a", "b")}
    with open(os.path.join(src, "README.md"), "w") as f:
        f.write(f"""# House Edge entry: 50/50 pool of two saved models

Recreate the forecast (run from this folder):

    python predict.py --data /data --out /output/submission.csv

- Expected output SHA-256: `{digest}`
- Model: P = {a.w:g} * A + {1 - a.w:g} * B over [p0, p1, p2, p5, p20] (weight fixed in advance, not tuned).
  - A = `member_a/` ({a.a_name}): its own code copy + `checkpoint.pkl` ({sizes['a']:.1f} MB).
  - B = `member_b/` ({a.b_name}): its own code copy + `checkpoint.pkl` ({sizes['b']:.1f} MB).
  Both were fitted on `<data>/train` (labels + histories) and the HISTORIES of validation and test (no validation answers),
  then saved. predict.py runs each member in its own Python process (they ship different versions of the same packages),
  then averages. No retraining at prediction time.
- Dependencies (`requirements.txt`, exact versions used to fit and pickle the checkpoints): {", ".join(pins)}.
  Built with Python {sys.version.split()[0]}.
- Parallelism: 2 workers, 2 BLAS/OpenMP threads per member; results do not depend on it. Local rerun time: {t_pred:.0f} s.
- Seeds: fixed everywhere; reruns are byte-identical. Writes only the `--out` file (plus temporary files in /tmp).
""")
    run([sys.executable, os.path.join(HERE, "kit", "check_submission.py"), csv_path, "--data", pub])
    zip_path = os.path.join(out, "entry.zip")
    run([sys.executable, os.path.join(HERE, "kit", "build_entry.py"), "--source", src, "--out", zip_path], cwd=os.path.join(HERE, "kit"))
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp); again = os.path.join(tmp, "rerun.csv")
        run([sys.executable, "predict.py", "--data", pub, "--out", again], cwd=tmp)
        if not filecmp.cmp(again, csv_path, shallow=False):
            sys.exit("RERUN MISMATCH: the ZIP does not reproduce the CSV")
    print(f"\nOK. Upload {csv_path} and {zip_path}. CSV SHA-256 {digest}. Rerun {t_pred:.0f} s, build {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
