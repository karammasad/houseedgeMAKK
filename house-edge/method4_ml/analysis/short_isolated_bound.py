"""Upper bound for using short logs' isolated older ticks. Long validation machines (stable/drift/shared), masked 3 ways:
 (a) last 40 only, (b) last 40 + 24 isolated older ticks (= short format), (c) same 24 ticks + their 8 preceding ticks revealed
(the most any missing-input marginalization could recover). P(win) MSE vs truth. Diagnosis only."""
import csv, random
from common.data import load_split
from method4_ml.model import Model
PUB = "data/main"
T = {r["query_id"]: 1 - float(r["p_0"]) for r in csv.DictReader(open(f"{PUB}/validation/targets.csv"))}
gm = Model(blend=0).fit(PUB).gm; q, seq = load_split(PUB, "validation")
rng = random.Random(0); err = {"a": 0.0, "b": 0.0, "c": 0.0}; n = 0
for r in q:
    s = seq[r["machine_id"]]; cut = r["cutoff"]
    if min(s) < 0: continue
    old = rng.sample(range(8, cut - 40), 24)
    keep = {"a": set(range(cut - 40, cut)), "b": set(range(cut - 40, cut)) | set(old),
            "c": set(range(cut - 40, cut)) | {t - d for t in old for d in range(0, 9)}}
    for k, ks in keep.items():
        sm = [x if i in ks else -1 for i, x in enumerate(s)]
        p = gm.win_probs([r], {r["machine_id"]: sm})[r["query_id"]]; err[k] += (p - T[r["query_id"]]) ** 2
    n += 1
print(f"{n} long machines masked; P(win) MSE vs truth:")
for k, lab in (("a", "last 40 only"), ("b", "last 40 + 24 isolated (short format)"), ("c", "+ isolated ticks' lag inputs revealed (ceiling)")):
    print(f"   ({k}) {lab:48s} {err[k] / n:.5f}")
