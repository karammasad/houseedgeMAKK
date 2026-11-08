import glob, pickle, random, sys
import numpy as np
sys.path.insert(0, "experiments/oracle")
from arms import ARMS
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "experiments/oracle/res[0-9].pkl")); R = [pickle.load(open(f, "rb")) for f in files]
P = np.concatenate([r["P"] for r in R]); meta = sum([r["meta"] for r in R], []); sp = R[0]["size_p"]
pt = np.array([m["p_true"] for m in meta]); s2 = np.array([1 + sum(x * x for x in sp[m["type"]]) for m in meta])   # 5-cat distance = (p-hat - p)^2 (1 + |s|^2) for a common size vector
E5 = (P - pt[:, None]) ** 2 * s2[:, None] * 1e4; EW = (P - pt[:, None]) ** 2 * 1e4
eps = np.array([m["ep"] for m in meta]); ueps = sorted(set(eps)); idx = {e: np.where(eps == e)[0] for e in ueps}
rng = np.random.default_rng(3); UE = {e: i for i, e in enumerate(ueps)}; EI = np.array([UE[e] for e in eps])
def ci(d, sel):
    ssum = np.bincount(EI[sel], weights=d[sel], minlength=len(ueps)); scnt = np.bincount(EI[sel], minlength=len(ueps)).astype(float)
    keep = scnt > 0; ssum, scnt = ssum[keep], scnt[keep]; pick = rng.integers(0, len(ssum), size=(2000, len(ssum)))
    ms = np.sort(ssum[pick].sum(1) / scnt[pick].sum(1)); return d[sel].mean(), ms[50], ms[1949]
print(f"{len(R)} worlds ({[f.split('res')[-1][:-4] for f in files]}), {len(meta)} test machines; 5-category squared distance to the TRUE distribution x1e-4 (common true size vector)")
subs = [("ALL", np.ones(len(meta), bool))] + [(t, np.array([m["type"] == t for m in meta])) for t in ("steady", "balanced", "jackpot")] + \
       [("long", np.array([not m["short"] for m in meta])), ("short", np.array([m["short"] for m in meta])),
        ("flipped", np.array([m["flipped"] and m["gated"] for m in meta])), ("no flip", np.array([(not m["flipped"]) and m["gated"] for m in meta])), ("no gate", np.array([not m["gated"] for m in meta])),
        ("GATED all", np.array([m["gated"] for m in meta])), ("GATED long", np.array([m["gated"] and not m["short"] for m in meta])), ("GATED short", np.array([m["gated"] and m["short"] for m in meta]))]
for name, sel in subs:
    print(f"\n[{name}] n={int(sel.sum())}")
    print(f"  {'arm':36s}{'error':>8s}{'vs A0 (paired) [95% CI]':>30s}{'vs B0':>28s}")
    for a, nm in enumerate(ARMS):
        m0 = ci(E5[:, a] - E5[:, 0], sel); mb = ci(E5[:, a] - E5[:, 3], sel)
        print(f"  {nm:36s}{E5[sel, a].mean():8.2f}{m0[0]:+10.2f} [{m0[1]:+.2f},{m0[2]:+.2f}]{mb[0]:+12.2f} [{mb[1]:+.2f},{mb[2]:+.2f}]")
