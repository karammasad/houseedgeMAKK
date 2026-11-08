"""Fit Adi's CURRENT adi.cook (r2 inner, adi size coupling, size menu; the same config as his pool) on the main kit and save its VALIDATION and TEST
predictions. Run from a scratch copy of origin/main (his files are not modified). Standard library only; no labels or targets are read by fit/predict."""
import pickle, sys, time
sys.path.insert(0, ".")
from common.data import public_dir, write_predictions
from adi.cook.model import Model as Cook
out = sys.argv[1]; pub = public_dir("data/main"); t0 = time.time()
m = Cook(inner="r2", size_couple="adi", workers=8).fit(pub)
pv = m.predict(pub, "validation"); pt = m.predict(pub, "test")
pickle.dump(pv, open(out + "/pred_cook_valid.pkl", "wb")); write_predictions(out + "/test_cook.csv", pt)
print(f"adi.cook fit + predict: {time.time() - t0:.0f}s; validation {len(pv)} rows, test {len(pt)} rows")
