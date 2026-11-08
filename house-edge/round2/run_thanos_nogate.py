"""Thanos E2 + the no-gate structure fix (mominsreverseengineering/experiments/rehearsal/nogate_fix.py), unmodified
monkeypatch of thanos.egp.screen. ONLY run this if fingerprint/probes on the real round-2 kit show a no-gate or
degenerate-rate (win rate >.97 or <.03) population — see round2/README.md decision rule. Mirrors thanos/run.py.

  python3 round2/run_thanos_nogate.py --data data/adaptation --out round2/results/thanos_nogate_r2 [--threads 2] [--also validation]
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/adaptation"); ap.add_argument("--out", required=True)
ap.add_argument("--splits", default="all", choices=("train", "all")); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tickset", default="own"); ap.add_argument("--also", default="validation")
A = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = str(A.threads)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mominsreverseengineering", "experiments", "rehearsal"))
import hashlib, json, pickle, resource, time  # noqa: E402
from common.data import public_dir, write_predictions  # noqa: E402
from thanos import egp  # noqa: E402
from thanos.model import Model  # noqa: E402
import nogate_fix  # noqa: E402

nogate_fix.apply(egp)
print("no-gate fix applied to thanos.egp.screen", flush=True)


def main():
    t0 = time.time(); c0 = time.process_time()
    pub = public_dir(A.data); os.makedirs(A.out, exist_ok=True)
    M = Model(A.splits, tickset=A.tickset).fit(pub)
    info = {"splits": A.splits, "threads": A.threads, "variant": "nogate_fix", "timing": {}}
    for sp in ["test"] + [s for s in A.also.split(",") if s]:
        P = M.predict(pub, sp)
        path = os.path.join(A.out, "submission.csv" if sp == "test" else f"{sp}.csv"); write_predictions(path, P)
        pickle.dump(P, open(os.path.join(A.out, f"{sp}.pkl"), "wb"))
        info[f"sha256_{sp}"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    info["timing"] = {k: round(v, 1) for k, v in M.timing.items()}
    info["wall_s"] = round(time.time() - t0, 1); info["cpu_s"] = round(time.process_time() - c0, 1)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    info["peak_rss_mb"] = round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    pickle.dump(M.H, open(os.path.join(A.out, "hyper.pkl"), "wb"))
    json.dump(info, open(os.path.join(A.out, "run.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
