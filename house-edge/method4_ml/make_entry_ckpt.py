"""Checkpoint entry builder (Person D): fit ONCE at build time, ship the fitted model, and have the organizer rerun only LOAD it and
predict the test split. Predictions are those of the fitted model, so a fresh fit and the shipped checkpoint agree exactly; the rerun
drops from a full refit (~18-27 min on 2 cores) to loading + test prediction. The kit guide endorses this ("Saving a trained checkpoint
is usually easier than retraining during the check"; "Load a saved checkpoint and stay within the runtime budget").

  python -m method4_ml.make_entry_ckpt --root mominsreverseengineering --kwargs '{"shared_base": true}' --data data/main --out build/main_sb_ckpt
  python -m method4_ml.make_entry_ckpt --data data/main --out build/main_v8_ckpt                       (plain v8 from this repo)

--root: the source tree whose packages are shipped (default: house-edge/). It is only READ: packages are copied into <out>/entry_src.
Steps: stage packages; FIT inside the staged copy (2 workers, 2 BLAS threads, cache off), write the fresh-fit test CSV, pickle the model;
run predict.py (load + predict); require it equals the fresh-fit CSV; kit checker; zip with kit/build_entry.py; unpack into a fresh
folder, rerun under the same thread limits, require agreement within 1e-8; record hashes, times and versions in verification.json.
"""
import argparse, csv, filecmp, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
THREADS = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}

FIT = '''"""Build-time only: fit the model and save the checkpoint (not run by the organizers)."""
import argparse, importlib, json, os, pickle, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common.data import public_dir, write_predictions
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--fresh-csv", required=True); a = ap.parse_args()
cfg = json.load(open(os.path.join(HERE, "entry_config.json"))); pub = public_dir(a.data)
kw = dict(cfg["kwargs"], workers=int(os.environ.get("FIT_WORKERS", cfg["kwargs"].get("workers", 2))))   # worker count never changes output
model = importlib.import_module(cfg["method"] + ".model").Model(**kw).fit(pub)
write_predictions(a.fresh_csv, model.predict(pub, "test"))     # predictions of the freshly fitted model
model.__dict__.pop("_sz", None)                                 # per-path prediction cache: not part of the fitted model
model.workers = cfg["kwargs"].get("workers", 2)                 # the organizer rerun uses the shipped setting (2 cores)
with open(os.path.join(HERE, "model.pkl"), "wb") as f:
    pickle.dump(model, f, protocol=4)
'''

PREDICT = '''"""Organizer entry point: python predict.py --data /data --out /output/submission.csv
Loads the fitted model (model.pkl, built from this round's public data by fit_checkpoint.py) and predicts /data/test."""
import argparse, os, pickle, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common.data import public_dir, write_predictions
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
with open(os.path.join(HERE, "model.pkl"), "rb") as f:
    model = pickle.load(f)
write_predictions(a.out, model.predict(public_dir(a.data), "test"))
'''


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True); t0 = time.time(); subprocess.run(cmd, check=True, **kw); return time.time() - t0


def read_probs(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh); assert rd.fieldnames == ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], rd.fieldnames
        return {r["query_id"]: [float(r[c]) for c in rd.fieldnames[1:]] for r in rd}


def maxdiff(A, B):
    assert A.keys() == B.keys(), "query ids differ"
    return max(abs(x - y) for k in A for x, y in zip(A[k], B[k]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=REPO); ap.add_argument("--method", default="method4_ml")
    ap.add_argument("--kwargs", default="{}"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); root = os.path.abspath(a.root); out = os.path.abspath(a.out); src = os.path.join(out, "entry_src")
    data = os.path.abspath(a.data); pub = os.path.join(data, "public") if os.path.isdir(os.path.join(data, "public")) else data
    if os.path.exists(out):
        sys.exit(f"{out} exists; choose a new --out folder")
    os.makedirs(src)
    pkgs = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "__init__.py")) and (d in ("common", "adi") or d.startswith("method")))
    for pkg in pkgs:
        os.makedirs(os.path.join(src, pkg))
        for f in os.listdir(os.path.join(root, pkg)):
            if f.endswith(".py"):
                shutil.copy(os.path.join(root, pkg, f), os.path.join(src, pkg, f))
    req = os.path.join(root, "method4_ml", "requirements.txt") if os.path.exists(os.path.join(root, "method4_ml", "requirements.txt")) else os.path.join(HERE, "requirements.txt")
    req_lines = [l.strip() for l in open(req) if l.strip() and not l.startswith("#")]
    import importlib.metadata as md
    mism = [f"{l}: installed {md.version(l.split('==')[0])}" for l in req_lines if md.version(l.split("==")[0]) != l.split("==")[1]]
    if mism:
        sys.exit("installed packages differ from the pins (a pickle must be built with the exact pinned versions):\n  " + "\n  ".join(mism))
    shutil.copy(req, os.path.join(src, "requirements.txt"))
    kw = {**json.loads(a.kwargs), "cache": False, "workers": 2}
    json.dump({"method": a.method, "kwargs": kw}, open(os.path.join(src, "entry_config.json"), "w"), indent=2)
    open(os.path.join(src, "fit_checkpoint.py"), "w").write(FIT); open(os.path.join(src, "predict.py"), "w").write(PREDICT)
    json.dump({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]},
              open(os.path.join(src, "reproduce.json"), "w"), indent=2)
    env = dict(os.environ, **THREADS)
    fresh = os.path.join(out, "fresh_fit_test.csv")
    fit_s = run([sys.executable, "fit_checkpoint.py", "--data", data, "--fresh-csv", fresh], cwd=src, env=dict(os.environ, FIT_WORKERS="8"))
    csv_path = os.path.join(out, "submission.csv")
    pred_s = run([sys.executable, "predict.py", "--data", data, "--out", csv_path], cwd=src, env=env)
    d_ck = maxdiff(read_probs(fresh), read_probs(csv_path))
    if d_ck > 1e-8:
        sys.exit(f"checkpoint predictions differ from the fresh fit: {d_ck:.3g}")
    digest = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    sh = lambda *c: subprocess.run(c, cwd=REPO, capture_output=True, text=True).stdout.strip()
    commit = sh("git", "rev-parse", "HEAD")
    open(os.path.join(src, "README.md"), "w").write(f"""# House Edge entry (checkpoint)

- Model: `{a.method}` from `{os.path.relpath(root, os.path.dirname(REPO))}`, settings `{json.dumps(kw, sort_keys=True)}` (entry_config.json)
- Source commit: {commit}
- Forecast SHA-256: {digest}
- What the command does: `python predict.py --data /data --out /output/submission.csv` LOADS the fitted model `model.pkl` and predicts
  `/data/test`. It does not retrain. Measured here: {pred_s:.0f} s with 2 workers / 2 BLAS threads.
- How model.pkl was made: `python fit_checkpoint.py --data <public> --fresh-csv <path>` (included) fits on this round's public data
  ({fit_s:.0f} s here). Its fresh-fit test CSV equals the checkpoint's output (max |diff| {d_ck:.1e}).
- Environment: image "python", Python 3.13 on the runner; requirements.txt pins the exact package versions the pickle was built with.
  Built with Python {sys.version.split()[0]}.
- Seeds: all randomness is seeded; same input -> byte-identical CSV.
- Data use: features use only observations strictly before each tick; missing ticks are unknown; fit/predict never read validation/test
  targets, labels or conditions.
""")
    run([sys.executable, os.path.join(REPO, "kit", "check_submission.py"), csv_path, "--data", pub])
    zip_path = os.path.join(out, "entry.zip")
    run([sys.executable, os.path.join(REPO, "kit", "build_entry.py"), "--source", src, "--out", zip_path], cwd=os.path.join(REPO, "kit"))
    names = zipfile.ZipFile(zip_path).namelist()
    for need in ("README.md", "reproduce.json", "requirements.txt", "predict.py", "entry_config.json", "model.pkl"):
        if need not in names:
            sys.exit(f"ZIP root is missing {need}")
    with open(os.path.join(pub, "test", "queries.csv"), newline="") as fh:
        want = sorted(r["query_id"] for r in csv.DictReader(fh))
    P = read_probs(csv_path); assert sorted(P) == want, "test queries missing or duplicated"
    assert all(all(math.isfinite(x) and x >= 0 for x in v) and abs(sum(v) - 1) < 1e-6 for v in P.values()), "invalid probabilities"
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp); again = os.path.join(tmp, "rerun.csv")
        rerun_s = run([sys.executable, "predict.py", "--data", data, "--out", again], cwd=tmp, env=env)
        same = filecmp.cmp(again, csv_path, shallow=False); d_re = maxdiff(P, read_probs(again))
        if not same and d_re > 1e-8:
            sys.exit(f"RERUN MISMATCH: {d_re:.3g}")
    info = dict(csv_sha256=digest, zip_sha256=hashlib.sha256(open(zip_path, "rb").read()).hexdigest(), zip_mb=round(os.path.getsize(zip_path) / 1e6, 2),
                commit=commit, root=root, settings=kw, python=sys.version.split()[0], packages={l.split("==")[0]: l.split("==")[1] for l in req_lines},
                fit_seconds=round(fit_s), predict_seconds=round(pred_s), rerun_seconds=round(rerun_s), checkpoint_vs_fresh_fit=d_ck,
                rerun_byte_identical=same, rerun_max_abs_diff=d_re, queries=len(P))
    json.dump(info, open(os.path.join(out, "verification.json"), "w"), indent=1)
    print("\nVERIFIED:", json.dumps({k: info[k] for k in ("fit_seconds", "predict_seconds", "rerun_seconds", "checkpoint_vs_fresh_fit", "rerun_byte_identical", "zip_mb")}))
    print(f"OK. Upload {csv_path} and {zip_path}. CSV SHA-256 {digest}")


if __name__ == "__main__":
    main()
