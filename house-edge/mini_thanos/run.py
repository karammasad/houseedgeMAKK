"""mini_thanos: routed entry. Long logs -> thanos (exact generator posterior); short logs -> the champion (Momin's shared-base, from its
saved checkpoint). The route uses only the observable log shape (fewer than half the ticks before the cutoff observed = short), never
condition labels. Why routed: see thanos/LOG.md sections 7-9 (thanos's long-log gain is confirmed on independent train outcomes, P .997;
its short-log gain is not).

    python3 -m mini_thanos.run --data data/main --out build/mini_thanos --ckpt build/main_sb_ckpt/entry_src [--also validation]

The champion runs in its own process from the checkpoint folder (that folder ships its own method4_ml / adi packages, which would shadow
this repo's). Records timing, peak RSS (this process and the champion child) and CSV hashes in <out>/run.json.
"""
import argparse, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/main"); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default="build/main_sb_ckpt/entry_src", help="champion checkpoint folder (model.pkl + its packages)")
ap.add_argument("--splits", default="all", choices=("train", "all"), help="thanos hyperparameters: train histories only, or all histories")
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--also", default="", help="extra splits to predict, e.g. validation")
A = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = str(A.threads)

import hashlib, json, pickle, resource, tempfile, time  # noqa: E402
from common.data import load_split, public_dir, write_predictions  # noqa: E402
from thanos.model import Model  # noqa: E402
from thanos import egp  # noqa: E402


def champion_predict(ckpt, pub, split):
    out = os.path.join(tempfile.mkdtemp(), "pred.pkl")
    code = ("import pickle,sys; sys.path.insert(0,'.'); m=pickle.load(open('model.pkl','rb')); "
            f"pickle.dump(m.predict({os.path.abspath(pub)!r},{split!r}), open({out!r},'wb'))")
    subprocess.run([sys.executable, "-c", code], cwd=os.path.abspath(ckpt), check=True, stdout=subprocess.DEVNULL)
    return pickle.load(open(out, "rb"))


def main():
    t0 = time.time(); pub = public_dir(A.data); os.makedirs(A.out, exist_ok=True)
    if not os.path.isfile(os.path.join(A.ckpt, "model.pkl")):
        sys.exit(f"champion checkpoint not found: {A.ckpt}/model.pkl (build it with method4_ml/make_entry_ckpt.py --root mominsreverseengineering)")
    M = Model(A.splits).fit(pub); t_fit = time.time() - t0
    info = {"thanos_splits": A.splits, "threads": A.threads, "ckpt": os.path.abspath(A.ckpt), "thanos_fit_s": round(t_fit, 1)}
    for sp in ["test"] + [s for s in A.also.split(",") if s]:
        q, seq = load_split(pub, sp); ms = egp.machines_from_split(q, seq)
        t1 = time.time(); C = champion_predict(A.ckpt, pub, sp); info[f"champion_predict_{sp}_s"] = round(time.time() - t1, 1)
        t1 = time.time(); long_ms = [m for m in ms if not m.short]; T = M.predict_machines(long_ms); info[f"thanos_predict_{sp}_s"] = round(time.time() - t1, 1)
        P = {m.qid: (C[m.qid] if m.short else T[m.qid]) for m in ms}
        assert len(P) == len(q) and all(abs(sum(v) - 1) < 1e-9 and min(v) >= 0 for v in P.values())
        info[f"routed_{sp}"] = {"thanos_long": len(long_ms), "champion_short": len(ms) - len(long_ms)}
        path = os.path.join(A.out, "submission.csv" if sp == "test" else f"{sp}.csv"); write_predictions(path, P)
        pickle.dump(P, open(os.path.join(A.out, f"{sp}.pkl"), "wb"))
        info[f"sha256_{sp}"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    info["wall_s"] = round(time.time() - t0, 1)
    k = 1024 * 1024 if sys.platform == "darwin" else 1024
    info["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / k, 1)
    info["peak_rss_child_mb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / k, 1)
    json.dump(info, open(os.path.join(A.out, "run.json"), "w"), indent=1); print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
