"""Entry builder for method4_ml (derived from Person D's builder; improved in mominsreverseengineering). Predictions are UNCHANGED by every change below.

What it does differently from the original:
  * CHECKPOINT: the model is fitted ONCE here (or reused with --model-pkl from adapt.py), pickled into the ZIP as model.pkl, and predict.py just loads it.
    The organizers' offline rerun then takes about a minute instead of ~27 min on 2 cores (limit: 30 min). If model.pkl is missing or cannot be loaded
    (e.g. a package-version mismatch) predict.py falls back to refitting from scratch, so the entry is never worse than before.
  * predict.py forces 2 pool workers and 2 BLAS/OpenMP threads (the organizer budget) instead of trusting os.cpu_count(), which a 2-core container may
    report as the host's core count.
  * requirements.txt is written from the packages actually installed here (the ones that fitted and pickled the model); the team's pin file is only
    compared and warned about (--strict-pins to fail). The old builder refused to run unless the installed stack equalled a stale pin file, and would
    otherwise ship pins that did not match the environment that produced the CSV.
  * the kit checker / ZIP builder are found in ROOT/kit or ROOT/../kit (the original assumed ROOT/kit, which does not exist in this folder).
  * verification: the CSV made by the packaged predict.py (loading the checkpoint) is compared with the CSV the in-memory fitted model gives, and a fresh
    unzip is rerun; both must agree within the organizers' 1e-8 (normally byte-identical).

Usage (from mominsreverseengineering/):
  python3 -m method4_ml.make_entry --data ../data/adaptation --out build/entry --kwargs '{"shared_base": true}'        # fits once, packages
  python3 -m method4_ml.make_entry --data ../data/adaptation --out build/entry --model-pkl build/r2/sb/model.pkl       # reuse a fitted model, no refit
  add --no-checkpoint for the original refit-in-the-rerun behaviour; --use-cache only for development (fast fit from caches).
Upload <out>/submission.csv and <out>/entry.zip.
"""
import argparse, filecmp, hashlib, json, os, pickle, shutil, subprocess, sys, tempfile, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_PKGS = ["adi"]   # non-method* packages the recommended model imports
PKG_NAMES = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
PREDICT = '''"""Organizer entry point: python predict.py --data /data --out /output/submission.csv
Loads the fitted model saved at build time (model.pkl); refits from scratch only if the checkpoint is missing or unreadable."""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")          # the organizer budget is two cores; do not trust os.cpu_count() inside a container
import argparse, importlib, json, pickle
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common.data import public_dir, write_predictions
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(os.path.join(HERE, "entry_config.json")) as f:
    cfg = json.load(f)
pub = public_dir(a.data); model = None; ck = os.path.join(HERE, "model.pkl")
if os.path.exists(ck):
    try:
        with open(ck, "rb") as f:
            model = pickle.load(f)
        model.cache = False; model.workers = 2
        print("loaded the fitted model from model.pkl", flush=True)
    except Exception as e:                  # e.g. package-version mismatch: fall back to a full refit, slower but identical in method
        print(f"WARNING: could not load model.pkl ({type(e).__name__}: {e}); refitting from scratch", flush=True); model = None
if model is None:
    kw = dict(cfg["kwargs"]); kw.setdefault("workers", 2)
    model = importlib.import_module(cfg["method"] + ".model").Model(**kw).fit(pub)
write_predictions(a.out, model.predict(pub, "test"))
'''


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)


def find_kit():
    for p in (os.path.join(ROOT, "kit"), os.path.join(os.path.dirname(ROOT), "kit")):
        if os.path.exists(os.path.join(p, "check_submission.py")):
            return p
    sys.exit("kit/ (check_submission.py, build_entry.py) not found next to or above this folder")


def read_probs(path):
    import csv as _csv
    with open(path, newline="") as fh:
        rd = _csv.DictReader(fh); assert rd.fieldnames == ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], rd.fieldnames
        return {r["query_id"]: [float(r[c]) for c in rd.fieldnames[1:]] for r in rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="method4_ml"); ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model-pkl", help="reuse an already fitted, pickled Model (e.g. from adapt.py) instead of fitting here")
    ap.add_argument("--no-checkpoint", action="store_true", help="original behaviour: the rerun refits from scratch")
    ap.add_argument("--use-cache", action="store_true", help="DEVELOPMENT ONLY: fit with the on-disk caches (fast)")
    ap.add_argument("--strict-pins", action="store_true", help="fail if the installed packages differ from method4_ml/requirements.txt")
    ap.add_argument("--allow-mismatch", action="store_true", help="(kept for compatibility; mismatch is now only a warning)")
    a = ap.parse_args()
    out = os.path.abspath(a.out); src = os.path.join(out, "entry_src"); kit = find_kit()
    if os.path.exists(out):
        sys.exit(f"{out} exists; choose a new --out folder")
    os.makedirs(src)
    pub = os.path.join(os.path.abspath(a.data), "public") if os.path.isdir(os.path.join(a.data, "public")) else os.path.abspath(a.data)
    for pkg in ["common"] + [p for p in EXTRA_PKGS if os.path.isdir(os.path.join(ROOT, p))] + sorted(d for d in os.listdir(ROOT) if d.startswith("method") and os.path.isdir(os.path.join(ROOT, d))):
        os.makedirs(os.path.join(src, pkg))
        for f in os.listdir(os.path.join(ROOT, pkg)):
            if f.endswith(".py"):
                shutil.copy(os.path.join(ROOT, pkg, f), os.path.join(src, pkg, f))
    # --- requirements: the packages that actually fit / pickled the model here
    import importlib.metadata as md
    installed = {n: md.version(n) for n in PKG_NAMES}
    pin_path = os.path.join(ROOT, "method4_ml", "requirements.txt"); mism = []
    if os.path.exists(pin_path):
        for l in (x.strip() for x in open(pin_path)):
            if l and not l.startswith("#") and "==" in l:
                n, v = l.split("==")
                if installed.get(n) != v:
                    mism.append(f"{n}: team pin file says {v}, installed {installed.get(n)}")
    if mism:
        print("WARNING: installed packages differ from method4_ml/requirements.txt (the entry ships what is INSTALLED, which is what produced the CSV):\n  " + "\n  ".join(mism))
        if a.strict_pins:
            sys.exit("--strict-pins: refusing to build")
    with open(os.path.join(src, "requirements.txt"), "w") as f:
        f.write("# exact versions of the packages that fitted and pickled this entry's model (see README)\n" + "".join(f"{n}=={v}\n" for n, v in installed.items()))
    kwargs = {**json.loads(a.kwargs), "cache": False}
    with open(os.path.join(src, "predict.py"), "w") as f:
        f.write(PREDICT)
    with open(os.path.join(src, "entry_config.json"), "w") as f:
        json.dump({"method": a.method, "kwargs": kwargs}, f, indent=2)
    with open(os.path.join(src, "reproduce.json"), "w") as f:
        json.dump({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, f, indent=2)
    # --- fit once (or reuse), checkpoint BEFORE any predict (so no data-path-keyed caches get pickled)
    sys.path.insert(0, ROOT); fit_s = None; direct = None; pkl_bytes = 0
    if not a.no_checkpoint:
        if a.model_pkl:
            with open(a.model_pkl, "rb") as fh:
                model = pickle.load(fh)
            print(f"reusing fitted model {a.model_pkl} (no refit)")
        else:
            import importlib
            t = time.time(); model = importlib.import_module(a.method + ".model").Model(**{**json.loads(a.kwargs), "cache": bool(a.use_cache)}).fit(pub); fit_s = time.time() - t
            print(f"fitted once in {fit_s:.0f}s")
        model.cache = False; model.workers = 2
        for k in ("_sz",):
            model.__dict__.pop(k, None)
        ck = os.path.join(src, "model.pkl")
        with open(ck, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_bytes = os.path.getsize(ck); print(f"checkpoint model.pkl: {pkl_bytes / 1e6:.1f} MB")
        if pkl_bytes > 90e6:
            sys.exit("model.pkl is too close to the 100 MB ZIP limit; use --no-checkpoint or shrink the model")
        direct = model.predict(pub, "test")        # what the in-memory fitted model predicts (compared with the packaged run below)
    csv_path = os.path.join(out, "submission.csv"); _t0 = time.time()
    run([sys.executable, "predict.py", "--data", pub, "--out", csv_path], cwd=src)
    runtime = time.time() - _t0
    digest = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    def sh(*c):
        try:
            return subprocess.run(c, cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "unknown"
    commit = sh("git", "rev-parse", "HEAD"); dirty = "yes" if sh("git", "status", "--porcelain", "--", ".") else "no"
    mode = "no-checkpoint" if a.no_checkpoint else "checkpoint"
    with open(os.path.join(src, "README.md"), "w") as f:
        f.write(f"""# House Edge entry (method4_ml)

- Method: `{a.method}`, settings `{json.dumps(kwargs, sort_keys=True)}` (entry_config.json). Mode: {mode}.
- Source commit: {commit} (uncommitted changes in this folder: {dirty})
- Forecast SHA-256: {digest}
- Built with: Python {sys.version.split()[0]}, {", ".join(f"{n} {v}" for n, v in installed.items())}; predict.py run measured {runtime:.0f} s on this machine
  ({'fit once at build time: ' + format(fit_s, '.0f') + ' s; ' if fit_s else ''}checkpoint {pkl_bytes / 1e6:.1f} MB).
- Environment: image "python"; third-party packages are pinned in requirements.txt (they are the ones that produced model.pkl).
  adi/, common/, method1_gate/, method2_sizes/, method4_ml/ are pure Python.
- Seeds: every random draw is seeded (training ticks per episode: SHA-256 of episode id + tick_seed; HistGradientBoosting
  random_state = 0..seeds-1; bags use tick_seed 0..bags-1). Same input -> byte-identical CSV.
- Limits: offline, no network, writes only the output CSV; predict.py sets 2 pool workers and 2 BLAS/OpenMP threads; peak memory far below 16 GiB.
- Data use: every feature for a query or training tick uses only observations strictly before that tick; missing ticks are unknown, not losses;
  fit/predict never read validation/test targets, labels or conditions (method4_ml/test_no_leakage.py; runtime file trace in experiments/audit_io.py).

Recreate: `python predict.py --data /data --out /output/submission.csv` (run from this folder). It loads `model.pkl` (the model fitted on this round's
training data and the histories of its validation/test splits) and predicts `/data/test`. If model.pkl cannot be loaded it refits from /data (slower, same method).
Training command (builds model.pkl and this ZIP): `python3 -m method4_ml.make_entry --data <kit public/> --out <dir> --kwargs '{json.dumps(json.loads(a.kwargs))}'`
""")
    run([sys.executable, os.path.join(kit, "check_submission.py"), csv_path, "--data", pub])
    zip_path = os.path.join(out, "entry.zip")
    run([sys.executable, os.path.join(kit, "build_entry.py"), "--source", src, "--out", zip_path], cwd=kit)
    names = zipfile.ZipFile(zip_path).namelist()
    for need in ("README.md", "reproduce.json", "requirements.txt", "predict.py", "entry_config.json") + (() if a.no_checkpoint else ("model.pkl",)):
        if need not in names:
            sys.exit(f"ZIP root is missing {need}: {sorted(n for n in names if '/' not in n)}")
    if any(n.startswith("/") or ".." in n.split("/") for n in names):
        sys.exit("ZIP contains unsafe paths")
    import csv as _csv, math
    with open(os.path.join(pub, "test", "queries.csv"), newline="") as fh:
        want = [r["query_id"] for r in _csv.DictReader(fh)]
    P = read_probs(csv_path)
    assert sorted(P) == sorted(want) and len(P) == len(want), "test queries missing or duplicated"
    for k, v in P.items():
        assert all(math.isfinite(x) and x >= 0 for x in v) and abs(sum(v) - 1) < 1e-6, f"invalid probabilities for {k}: {v}"
    direct_diff = None
    if direct is not None:
        direct_diff = max(abs(x - y) for k in P for x, y in zip(P[k], direct[k]))
        if direct_diff > 1e-8:
            sys.exit(f"CHECKPOINT MISMATCH: packaged predict.py differs from the in-memory model by {direct_diff:.3g} (> 1e-8)")
    with tempfile.TemporaryDirectory() as tmp:         # fresh unzip + rerun, exactly as the organizers will
        zipfile.ZipFile(zip_path).extractall(tmp)
        again = os.path.join(tmp, "rerun.csv"); _t1 = time.time()
        run([sys.executable, "predict.py", "--data", pub, "--out", again], cwd=tmp)
        rerun_s = time.time() - _t1
        same_bytes = filecmp.cmp(again, csv_path, shallow=False)
        Q = read_probs(again); maxdiff = max(abs(x - y) for k in P for x, y in zip(P[k], Q[k]))
        if not same_bytes and maxdiff > 1e-8:
            sys.exit(f"RERUN MISMATCH: max |diff| {maxdiff:.3g} exceeds the organizer tolerance 1e-8")
    info = dict(mode=mode, csv_sha256=digest, zip_sha256=hashlib.sha256(open(zip_path, "rb").read()).hexdigest(), zip_mb=round(os.path.getsize(zip_path) / 1e6, 1),
                commit=commit, dirty=dirty, python=sys.version.split()[0], platform=sys.platform, settings=kwargs, packages=installed, pin_file_mismatches=mism,
                fit_seconds=None if fit_s is None else round(fit_s), packaged_predict_seconds=round(runtime), rerun_seconds=round(rerun_s), byte_identical=same_bytes,
                max_abs_diff_rerun=maxdiff, max_abs_diff_vs_in_memory_model=direct_diff, queries=len(P), model_pkl_mb=round(pkl_bytes / 1e6, 1),
                zip_root=sorted(n for n in names if "/" not in n))
    with open(os.path.join(out, "verification.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print("\nVERIFIED:", json.dumps({k: info[k] for k in ("mode", "fit_seconds", "packaged_predict_seconds", "rerun_seconds", "byte_identical", "max_abs_diff_rerun", "max_abs_diff_vs_in_memory_model", "zip_mb", "queries")}))
    print(f"OK. Upload {csv_path} and {zip_path}. CSV SHA-256 {digest}")


if __name__ == "__main__":
    main()
