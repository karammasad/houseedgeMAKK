"""(5) disabled-structure equality vs the ORIGINAL gate (git snapshot 6f8decd) and (6) fit the shared-base structure on a copy of the baseline gate."""
import copy, importlib.util, json, math, subprocess, sys, time
import numpy as np
sys.path.insert(0, ".")
from common.data import public_dir, load_split
from adi import gate as G
from method4_ml.model import Model
from method4_ml.pooled import pooled_split
from method4_ml.shared_base import attach_shared_base

pub = public_dir("../data/main"); gm = Model(cache=True)._fit_gate(pub)           # baseline gate (cache hit)
src = subprocess.run(["git", "show", "6f8decd:house-edge/mominsreverseengineering/adi/gate.py"], capture_output=True, text=True, cwd="..").stdout
spec = importlib.util.spec_from_loader("gate_orig", loader=None); orig = importlib.util.module_from_spec(spec)
orig.__dict__["__name__"] = "gate_orig"; exec(compile(src, "gate_orig", "exec"), orig.__dict__)
worst = 0.0; n = 0
for split in ("validation", "test"):
    q, seq = load_split(pub, split)
    for r in q:
        s = seq[r["machine_id"]]; cfg = gm._mcfg(s, r_type=r["machine_type"])
        a = G.win_probability(r, s, gm.ST, gm._sp(s), cellprior=gm._cp(r), **cfg)
        b = orig.win_probability(r, s, gm.ST, gm._sp(s), cellprior=gm._cp(r), **cfg)
        worst = max(worst, abs(a - b)); n += 1
print(f"(5) DISABLED-STRUCTURE EQUALITY: current gate vs original gate on {n} validation+test queries: max |P(win) diff| = {worst:.2e}", flush=True)
t0 = time.time(); g2 = copy.deepcopy(gm); qs, sq = pooled_split(pub, ("train",))
info = attach_shared_base(g2, qs, sq)
print(f"(6) attach_shared_base: {time.time() - t0:.0f}s")
print("derived pairs (pair, size lag):", info["pairs"])
for t, d in info["per_type"].items():
    print(f"  {t:9s} {d}")
print("structure prior after relearning:", info["structure_prior"])
json.dump(info, open("experiments/shared_fit_info.json", "w"), indent=1, default=float)
import pickle; pickle.dump(g2, open("experiments/gm_sb.pkl", "wb"))
