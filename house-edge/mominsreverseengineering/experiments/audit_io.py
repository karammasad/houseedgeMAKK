"""Runtime I/O audit: which files under the kit does fit() / predict() actually open? (Python audit hook; inherited by forked workers.)
Run: python3 experiments/audit_io.py <kit public dir>.  Opens are appended to a log; the summary groups them by split and file name."""
import collections, json, os, sys
pub = os.path.abspath(sys.argv[1]); LOG = os.path.abspath("experiments/audit_io.log"); open(LOG, "w").close()
def hook(event, args):
    if event == "open":
        p = args[0]
        if isinstance(p, bytes): p = p.decode(errors="ignore")
        if isinstance(p, str):
            ap = os.path.abspath(p)
            if ap.startswith(pub + os.sep):
                with open(LOG, "a") as f:
                    f.write(f"{os.getpid()}\t{'fit' if STATE['phase'] == 'fit' else STATE['phase']}\t{os.path.relpath(ap, pub)}\n")
STATE = {"phase": "fit"}
sys.addaudithook(hook)
sys.path.insert(0, ".")
from method4_ml.model import Model
m = Model(cache=False, shared_base=True, workers=4).fit(pub)
STATE["phase"] = "predict_validation"; m.predict(pub, "validation")
STATE["phase"] = "predict_test"; m.predict(pub, "test")
c = collections.Counter(); procs = collections.defaultdict(set)
for line in open(LOG):
    pid, phase, rel = line.rstrip("\n").split("\t"); c[(phase, rel)] += 1; procs[(phase, rel)].add(pid)
print("FILES OPENED UNDER THE KIT (phase, file: opens, distinct processes):")
for (phase, rel), n in sorted(c.items()):
    print(f"  {phase:20s} {rel:32s} {n:4d}  {len(procs[(phase, rel)])}")
forbidden = [k for k in c if os.path.basename(k[1]) in ("labels.csv", "targets.csv", "conditions.csv", "reference.csv", "sample_submission.csv")]
print("\nTARGET-BEARING FILES OPENED:", forbidden or "NONE")
