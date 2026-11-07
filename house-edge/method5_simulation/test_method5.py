"""Unit tests for Method 5 (python -m unittest method5_simulation.test_method5). Skips if main-round data is absent."""
import math, os, random, unittest
import numpy as np
from common.data import load_split
from method5_simulation import data as D
from method5_simulation.model import Model, featurize, L

DATA = os.environ.get("M5_TEST_DATA", "data/main")


@unittest.skipUnless(os.path.isdir(os.path.join(DATA, "validation")), "main-round data not present")
class TestMethod5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q, cls.seq = load_split(DATA, "validation")

    def test_stdlib_features_match_numpy_builder(self):
        rng = random.Random(0)
        for r in rng.sample(self.q, 40):
            s = self.seq[r["machine_id"]]; t = r["cutoff"]
            w, S = featurize(s, t, r["machine_type"], int(r["signal"]))
            W2, S2 = D.build(D.machine_cats(s), [t], r["machine_type"], int(r["signal"]))
            self.assertEqual(w, W2[0].tolist()); self.assertLess(max(abs(a - b) for a, b in zip(S, S2[0])), 1e-6)

    def test_features_use_only_earlier_ticks(self):
        rng = random.Random(1)
        for r in rng.sample(self.q, 25):
            s = list(self.seq[r["machine_id"]]); t = rng.randrange(80, len(s))
            a = featurize(s, t, r["machine_type"], 0); s2 = s[:t] + [rng.choice((-1, 0, 1, 2, 5, 20)) for _ in s[t:]]
            self.assertEqual(a, featurize(s2, t, r["machine_type"], 0))

    def test_unobserved_ticks_are_flagged_never_losses(self):
        r = next(r for r in self.q if (np.array(self.seq[r["machine_id"]]) < 0).mean() > .5)
        w, _ = featurize(self.seq[r["machine_id"]], r["cutoff"], r["machine_type"], 0)
        self.assertIn(-1, w); self.assertEqual(w.count(-1), sum(1 for v in self.seq[r["machine_id"]][-L:] if v < 0))

    def test_nogap_window_keeps_order_and_right_aligns_observed(self):
        w = [-1, 2, -1, 0, 1] + [-1] * (L - 7) + [3 if False else 4, -1]
        obs = [c for c in w if c >= 0]; ng = [-1] * (L - len(obs)) + obs
        self.assertEqual(ng[-4:], [2, 0, 1, 4]); self.assertEqual(len(ng), L)
        order = np.argsort(np.array(w) >= 0, kind="stable"); self.assertEqual(np.array(w)[order].tolist(), ng)

    def test_predictions_valid_and_deterministic(self):
        for tag in ("s1", "s2"):
            path = f"method5_simulation/weights/{tag}.json"
            if not os.path.exists(path):
                continue
            m = Model(path).fit(DATA); r = self.q[0]; s = self.seq[r["machine_id"]]
            p = m.forward(*featurize(s, r["cutoff"], r["machine_type"], int(r["signal"])))
            self.assertEqual(len(p), 5); self.assertTrue(all(x >= 0 for x in p)); self.assertAlmostEqual(sum(p), 1.0, places=12)
            self.assertEqual(p, Model(path).fit(DATA).forward(*featurize(s, r["cutoff"], r["machine_type"], int(r["signal"]))))


if __name__ == "__main__":
    unittest.main()
