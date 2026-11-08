"""2-core rerun timing of a SAVED, fitted adi.cook (what an organizer rerun of a checkpoint entry does for this member): fresh process, 2 BLAS/OpenMP threads,
2 workers, load the pickle, predict TEST, hard kill at 29 minutes. Reports wall, CPU-seconds, peak memory (MB) and the difference vs the earlier test CSV."""
import os, sys, time, signal, resource
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "2"
os.setpgrp(); signal.signal(signal.SIGALRM, lambda s, f: (print("TIMEOUT at 29 minutes"), os.killpg(0, signal.SIGKILL))); signal.alarm(29 * 60)
sys.path.insert(0, "."); import pickle, csv
pub = os.path.abspath(sys.argv[1]); outdir = sys.argv[2]; t0 = time.time()
m = pickle.load(open(os.path.join(outdir, "cook_model.pkl"), "rb")); t1 = time.time()
for o in (m, getattr(m, "inner", None)):
    if o is not None and hasattr(o, "workers"): o.workers = 2
pt = m.predict(pub, "test"); t2 = time.time(); signal.alarm(0)
ref = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in csv.DictReader(open(os.path.join(outdir, "test_cook.csv")))}
d = max(abs(a - b) for q in ref for a, b in zip(ref[q], pt[q])); t = os.times()
print(f"saved cook, 2 workers / 2 threads: load {t1 - t0:.1f}s + predict(test) {t2 - t1:.1f}s = {t2 - t0:.1f}s wall; CPU-seconds {t.user + t.system + t.children_user + t.children_system:.0f}; "
      f"peak memory self {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB, largest worker {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6:.0f} MB; max |diff| vs earlier test CSV {d:.2e}")
