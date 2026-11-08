"""Method 4's entry builder (Person D), derived from method2_sizes/make_entry.py. Differences: packages adi/, writes a
requirements.txt with the EXACT installed scikit-learn stack, and forces cache=false (the submission folder is read-only).

Why: tools/make_entry.py stages only common/ and method*/ folders, so any entry that uses Adi's gate
(e.g. method2_sizes with win_model="adi") fails on the runner with "No module named 'adi'".
The shared tools/make_entry.py is left untouched; this copy lives in method2_sizes/ only.

Usage (from the repo root):
  python -m method4_ml.make_entry --data data/main --out build/main_m4   (add --kwargs to override model settings)

Steps: stage common/ and all method*/ packages plus a generated predict.py, README.md, reproduce.json and
entry_config.json; predict the test split; run the kit checker; zip with kit/build_entry.py; unpack the ZIP
into a fresh folder, rerun it, and require a byte-identical CSV. Upload <out>/submission.csv and <out>/entry.zip.
"""
import argparse, filecmp, hashlib, json, os, shutil, subprocess, sys, tempfile, time, zipfile

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
    ap.add_argument("--method", default="method4_ml"); ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--allow-mismatch", action="store_true")
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
    req_path = os.path.join(ROOT, "method4_ml", "requirements.txt")
    req_lines = [l.strip() for l in open(req_path) if l.strip() and not l.startswith("#")]
    import importlib.metadata as md
    mism = []
    for l in req_lines:
        name, ver = l.split("==")
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        if have != ver:
            mism.append(f"{name}: requirements.txt pins {ver}, installed {have}")
    if mism and not a.allow_mismatch:
        sys.exit("requirements.txt does not match the environment used to build/verify this entry:\n  " + "\n  ".join(mism) +
                 "\nFix method4_ml/requirements.txt (or pass --allow-mismatch knowingly).")
    shutil.copy(req_path, os.path.join(src, "requirements.txt"))
    with open(os.path.join(src, "predict.py"), "w") as f:
        f.write(PREDICT)
    with open(os.path.join(src, "entry_config.json"), "w") as f:
        json.dump({"method": a.method, "kwargs": {**json.loads(a.kwargs), "cache": False}}, f, indent=2)
    with open(os.path.join(src, "reproduce.json"), "w") as f:
        json.dump({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, f, indent=2)
    csv_path = os.path.join(out, "submission.csv")
    import time; _t0 = time.time()
    run([sys.executable, "predict.py", "--data", os.path.abspath(a.data), "--out", csv_path], cwd=src)
    runtime = time.time() - _t0
    digest = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    def sh(*c):
        try:
            return subprocess.run(c, cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "unknown"
    commit = sh("git", "rev-parse", "HEAD"); dirty = "yes" if sh("git", "status", "--porcelain", "--", "house-edge") else "no"
    with open(os.path.join(src, "README.md"), "w") as f:
        f.write(f"""# House Edge entry (Method 4, method4_ml)

- Method: `{a.method}`, settings `{json.dumps({**json.loads(a.kwargs), "cache": False}, sort_keys=True)}` (entry_config.json)
- Source commit: {commit} (uncommitted changes in house-edge/: {dirty})
- Forecast SHA-256: {digest}
- Built with: Python {sys.version.split()[0]}; measured fit + test predict: {runtime:.0f} s on this machine with the settings above
- Environment: image "python"; the code needs the third-party packages pinned in requirements.txt (scikit-learn stack, installed
  by the organizers before the offline run). adi/, common/, method1_gate/, method2_sizes/, method4_ml/ are pure Python.
- Seeds: every random draw is seeded (training ticks per episode: SHA-256 of episode id + tick_seed; HistGradientBoosting
  random_state = 0..seeds-1; bags use tick_seed 0..bags-1). Same input -> byte-identical CSV.
- Limits: runs offline, no network, writes only the output CSV (cache disabled: the submission folder is read-only);
  `workers` processes are used for feature building; peak memory well under 16 GiB.
- Data use: every feature for a query or training tick uses only observations strictly before that tick; missing ticks are
  unknown, not losses; fit/predict never read validation/test targets, labels or conditions (method4_ml/test_no_leakage.py).

Recreate: `python predict.py --data /data --out /output/submission.csv` (run from this folder). It refits from `/data/train`
(+ histories of /data/validation and /data/test, allowed) and predicts `/data/test`. No saved checkpoint is needed.
""")
    pub = os.path.join(os.path.abspath(a.data), "public") if os.path.isdir(os.path.join(a.data, "public")) else os.path.abspath(a.data)
    run([sys.executable, os.path.join(ROOT, "kit", "check_submission.py"), csv_path, "--data", pub])
    zip_path = os.path.join(out, "entry.zip")
    run([sys.executable, os.path.join(ROOT, "kit", "build_entry.py"), "--source", src, "--out", zip_path], cwd=os.path.join(ROOT, "kit"))
    # --- ZIP layout: must open directly to README.md and reproduce.json; requirements.txt must be at the ROOT (external packages)
    names = zipfile.ZipFile(zip_path).namelist()
    for need in ("README.md", "reproduce.json", "requirements.txt", "predict.py", "entry_config.json"):
        if need not in names:
            sys.exit(f"ZIP root is missing {need}: {sorted(n for n in names if '/' not in n)}")
    if any(n.startswith("/") or ".." in n.split("/") for n in names):
        sys.exit("ZIP contains unsafe paths")
    # --- CSV validity: every test query once, finite, nonnegative, sums to 1 (independent of the kit checker)
    import csv as _csv, math
    with open(os.path.join(pub, "test", "queries.csv"), newline="") as fh:
        want = [r["query_id"] for r in _csv.DictReader(fh)]
    def read_probs(path):
        with open(path, newline="") as fh:
            rd = _csv.DictReader(fh); assert rd.fieldnames == ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], rd.fieldnames
            return {r["query_id"]: [float(r[c]) for c in rd.fieldnames[1:]] for r in rd}
    P = read_probs(csv_path)
    assert sorted(P) == sorted(want) and len(P) == len(want), "test queries missing or duplicated"
    for k, v in P.items():
        assert all(math.isfinite(x) and x >= 0 for x in v) and abs(sum(v) - 1) < 1e-6, f"invalid probabilities for {k}: {v}"
    # --- fresh unzip + rerun (this is also the environment check: it runs with whatever interpreter/packages launched this builder)
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp)
        again = os.path.join(tmp, "rerun.csv"); _t1 = time.time()
        run([sys.executable, "predict.py", "--data", os.path.abspath(a.data), "--out", again], cwd=tmp)
        rerun_s = time.time() - _t1
        same_bytes = filecmp.cmp(again, csv_path, shallow=False)
        Q = read_probs(again); maxdiff = max(abs(x - y) for k in P for x, y in zip(P[k], Q[k]))
        if not same_bytes and maxdiff > 1e-8:
            sys.exit(f"RERUN MISMATCH: max |diff| {maxdiff:.3g} exceeds the organizer tolerance 1e-8")
    import importlib.metadata as md
    info = dict(csv_sha256=digest, zip_sha256=hashlib.sha256(open(zip_path, "rb").read()).hexdigest(), commit=commit, dirty=dirty,
                python=sys.version.split()[0], platform=sys.platform, settings={**json.loads(a.kwargs), "cache": False},
                packages={n.split("==")[0]: md.version(n.split("==")[0]) for n in req_lines}, first_run_seconds=round(runtime),
                rerun_seconds=round(rerun_s), byte_identical=same_bytes, max_abs_diff=maxdiff, queries=len(P),
                zip_root=sorted(n for n in names if "/" not in n))
    with open(os.path.join(out, "verification.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print("\nVERIFIED:", json.dumps({k: info[k] for k in ("python", "first_run_seconds", "rerun_seconds", "byte_identical", "max_abs_diff", "queries")}))
    print(f"OK. Upload {csv_path} and {zip_path}. CSV SHA-256 {digest}")

if __name__ == "__main__":
    main()
