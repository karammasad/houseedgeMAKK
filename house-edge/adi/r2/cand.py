"""Fit one runbook candidate in its own process; write validation + test predictions and a learned-summary text.

  python -m adi.r2.cand --method adi.cook --kwargs '{...}' --data <pub> --out <dir>
Separate processes keep each candidate's in-process patches (adi.cook) away from the others."""
import argparse, contextlib, importlib, io, json, os, sys, time
from common.data import public_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); pub = public_dir(a.data); t0 = time.time()
    m = importlib.import_module(a.method + ".model").Model(**json.loads(a.kwargs)).fit(pub)
    res = {"validation": m.predict(pub, "validation") if os.path.exists(os.path.join(pub, "validation", "queries.csv")) else {},
           "test": m.predict(pub, "test"), "seconds": time.time() - t0}
    with open(os.path.join(a.out, "pred.json"), "w") as f:
        json.dump(res, f)
    try:
        from adi.r2.run import describe
        txt = describe(m)
    except Exception as e:  # the summary is informational only
        txt = f"(no summary: {e!r})"
    with open(os.path.join(a.out, "learned.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"done {a.method} {time.time() - t0:.0f}s")
