"""ONE representative deployment test: what the organizers' offline rerun does (fit from scratch, cache OFF, predict TEST, write CSV), with the
organizers' 2-core budget emulated: 2 pool workers, 2 BLAS / OpenMP threads, hard kill at 29 minutes. macOS has no CPU-affinity API, so cores are
limited by thread / worker counts (an emulation; CPU-seconds are reported so wall time can be bounded on any 2-core machine).
Per-stage timers wrap the model's own functions; no model code or maths is changed.   Run: python3 experiments/deploy_test.py [shared_base|base]"""
import os, sys, time, signal, resource
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "2"
os.setpgrp(); sys.path.insert(0, "."); T0 = time.time(); STAGE = ["start"]; TIMES = {}
LIMIT = 29 * 60
def report(tag):
    t = os.times(); ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; rc = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(f"\n[{tag}] wall {time.time() - T0:.0f}s | stage when stopped/finished: {STAGE[0]}")
    for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1]): print(f"   {k:34s} {v:7.0f}s")
    print(f"   CPU-seconds (user+sys, self {t.user + t.system:.0f}, children {t.children_user + t.children_system:.0f}); peak RSS self {ru / 1e6:.2f} GB, largest child {rc / 1e6:.2f} GB", flush=True)
def on_alarm(sig, frm):
    report("TIMEOUT at 29 minutes"); os.killpg(0, signal.SIGKILL)
signal.signal(signal.SIGALRM, on_alarm); signal.alarm(LIMIT)
import threadpoolctl; threadpoolctl.threadpool_limits(2)
import method4_ml.model as M
from common.data import public_dir, write_predictions
def timed(name, fn):
    def w(*a, **k):
        STAGE[0] = name; t = time.time()
        try: return fn(*a, **k)
        finally: TIMES[name] = TIMES.get(name, 0) + time.time() - t
    return w
M.attach_shared_base = timed("shared-base structure fit", M.attach_shared_base)
M.fit_menu = timed("size menu fit", M.fit_menu)
M.Model._fit_gate = timed("gate fit (incl. signal shift, sb)", M.Model._fit_gate)
M.Model._fit_sizes = timed("size model fit", M.Model._fit_sizes)
M.Model._features = timed("feature building (all splits)", M.Model._features)
mode = sys.argv[1] if len(sys.argv) > 1 else "shared_base"
kw = dict(cache=False, workers=2, **({"shared_base": True} if mode == "shared_base" else {}))
pub = public_dir("../data/main"); m = M.Model(**kw)
t = time.time(); m.fit(pub); fit_s = time.time() - t; STAGE[0] = "predict test"
t = time.time(); pred = m.predict(pub, "test"); pred_s = time.time() - t
TIMES["fit total"] = fit_s; TIMES["predict test"] = pred_s
out = f"experiments/test_{mode}_2core.csv"; write_predictions(out, pred); STAGE[0] = "done"; signal.alarm(0)
report(f"FINISHED {mode} within the limit" if time.time() - T0 < LIMIT else "FINISHED late")
if mode == "shared_base" and os.path.exists("experiments/test_sb.csv"):
    import csv
    ref = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in csv.DictReader(open("experiments/test_sb.csv"))}
    d = max(abs(x - y) for q, v in pred.items() for x, y in zip(v, ref[q]))
    print(f"max abs probability difference vs the 8-thread test CSV (allowed by the organizers: 1e-8): {d:.2e}")
