import sys, time, pickle, os
sys.path.insert(0, "experiments/oracle")
import sim
from adi.model import Model as AdiModel
from method4_ml.model import GATE
from method4_ml.signal_shift import attach_signal_shift
from method4_ml.shared_base import attach_shared_base
from method4_ml.pooled import pooled_split


def fit_world(world, workdir):
    pub = os.path.join(workdir, "pub"); sim.write_pub(world["train"], pub); t0 = time.time()
    gm = AdiModel(**GATE).fit(pub); t1 = time.time()
    attach_signal_shift(gm, *pooled_split(pub, ("train",))); t2 = time.time()
    attach_shared_base(gm, *pooled_split(pub, ("train",))); t3 = time.time()
    return gm, dict(adi=t1 - t0, signal=t2 - t1, shared=t3 - t2)

if __name__ == "__main__":
    t0 = time.time(); w = sim.make_world(int(sys.argv[1]) if len(sys.argv) > 1 else 1); print("world", round(time.time() - t0, 1), "s")
    gm, tm = fit_world(w, "experiments/oracle/work1"); print(tm); print(len(gm.ST), "structures")
    names = [f"{f}{l}" for f, l in gm.ST] + ["none"]
    for n, p in sorted(zip(names, gm.sprior), key=lambda x: -x[1])[:10]: print(f"  {n:24s} {p:.3f}")
    pickle.dump(gm, open("experiments/oracle/work1/gm.pkl", "wb"))
