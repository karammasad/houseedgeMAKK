"""Leakage check for adi.pool (analysis): fit + predict on a copy of the kit with EVERY answer file deleted (validation labels/
targets/conditions/reference, train conditions); log every file opened (inherited by forked workers). The predictions must be
identical to the clean run's, and no answer file may be opened."""
import builtins, io, json, os, sys
LOG = sys.argv[3]; _open = builtins.open
def logged(f, *a, **k):
    try:
        with _open(LOG, "a") as g:
            g.write(f"{f}\n")
    except Exception:
        pass
    return _open(f, *a, **k)
builtins.open = io.open = logged
sys.path.insert(0, os.getcwd())
from adi.pool.model import Model
m = Model().fit(sys.argv[1])
json.dump({s: m.predict(sys.argv[1], s) for s in ("validation", "test")}, _open(sys.argv[2], "w"))
print("done")
