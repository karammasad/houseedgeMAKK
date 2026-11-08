"""Leakage + reproducibility check of Adi's current adi.cook, plus a saved fitted model for the 2-core timing.
Run from a scratch copy of origin/main with the ANSWER-FREE kit: only queries.csv and history.csv of train/validation/test exist (no labels, targets,
conditions, reference, sample_submission). A Python audit hook logs every file opened under the kit, including forked workers. If the fit/predict needs an answer
file it crashes; if it ignores them, validation and test predictions equal the run on the full kit (max difference reported)."""
import collections, os, pickle, sys, time
pub = os.path.abspath(sys.argv[1]); outdir = sys.argv[2]; LOG = os.path.join(outdir, "leak_cook_opens.log"); open(LOG, "w").close()
def hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        p = os.path.abspath(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if p.startswith(pub + os.sep):
            with open(LOG, "a") as f: f.write(f"{os.getpid()}\t{os.path.relpath(p, pub)}\n")
sys.addaudithook(hook); sys.path.insert(0, ".")
from common.data import write_predictions
from adi.cook.model import Model as Cook
t0 = time.time(); m = Cook(inner="r2", size_couple="adi", workers=8).fit(pub)
pv = m.predict(pub, "validation"); pt = m.predict(pub, "test"); fit_s = time.time() - t0
pickle.dump(m, open(os.path.join(outdir, "cook_model.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
ref = pickle.load(open(os.path.join(outdir, "pred_cook_valid.pkl"), "rb"))
dv = max(abs(a - b) for q in ref for a, b in zip(ref[q], pv[q]))
import csv
rt = {r["query_id"]: [float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in csv.DictReader(open(os.path.join(outdir, "test_cook.csv")))}
write_predictions(os.path.join(outdir, "test_cook_noans.csv"), pt)
dt = max(abs(a - b) for q in rt for a, b in zip(rt[q], pt[q]))
c = collections.Counter(); procs = collections.defaultdict(set)
for line in open(LOG):
    pid, rel = line.rstrip("\n").split("\t"); c[rel] += 1; procs[rel].add(pid)
print(f"cook fit + validation + test predict on the ANSWER-FREE kit: {fit_s:.0f}s")
print("FILES OPENED UNDER THE KIT (opens, distinct processes):")
for rel, n in sorted(c.items()): print(f"  {rel:28s} {n:4d} {len(procs[rel])}")
print(f"max |diff| validation predictions vs the run on the FULL kit: {dv:.2e}; test predictions vs full-kit run: {dt:.2e}")
