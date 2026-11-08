"""End-to-end reproduction path (what the organizer rerun would execute): load data, fit every hyperparameter, predict, write CSVs.
    python -m thanos_method.run --data data/main --splits train --out build/egp_ind [--threads 2] [--also validation]
Records wall / CPU time per stage, peak RSS and the CSV SHA-256 in <out>/run.json. Threads are pinned BEFORE numpy is imported.
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/main"); ap.add_argument("--out", required=True)
ap.add_argument("--splits", default="train", choices=("train", "all")); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tickset", default="own"); ap.add_argument("--also", default="", help="comma list of extra splits to predict (e.g. validation)")
A = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = str(A.threads)

import hashlib, json, pickle, resource, time  # noqa: E402
from common.data import public_dir, write_predictions  # noqa: E402
from .model import Model  # noqa: E402


def main():
    t0 = time.time(); c0 = time.process_time()
    pub = public_dir(A.data); os.makedirs(A.out, exist_ok=True)
    M = Model(A.splits, tickset=A.tickset).fit(pub)
    info = {"splits": A.splits, "threads": A.threads, "timing": {}}
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
