"""Fast unit tests for Method 2's size model (standard library unittest). Run from the repo root:

  python -m unittest method2_sizes.test_method2 -v

Uses a train subset so the whole file runs in well under a minute. Full-pipeline reproducibility (byte-identical
CSV, valid submission) is checked separately by method2_sizes/make_entry.py.
"""
import math, os, unittest
from common.data import load_split

DATA = os.environ.get("M2_TEST_DATA", "data/main")


def _subset(n=160):
    q, seq = load_split(DATA, "train")
    eps = sorted({r["episode_id"] for r in q})[: n // 8]
    q = [r for r in q if r["episode_id"] in eps]
    return q, {r["machine_id"]: seq[r["machine_id"]] for r in q}


def _model(**kw):
    from method2_sizes.sizes import GateSizes
    base = dict(cell_prior="npmle", couple=True, couple_iters=5, tied=True, tie_weight="learn",
                joint_em=True, joint_iters=8)
    base.update(kw)
    return GateSizes(**base)


@unittest.skipUnless(os.path.isdir(os.path.join(DATA, "train")), "main-round data not present")
class TestGateSizes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q, cls.seq = _subset()
        cls.base = _model().fit(cls.q, cls.seq)
        cls.dirm = _model(learn_direction=True).fit(cls.q, cls.seq)

    def _preds(self, m):
        return {r["query_id"]: m.predict(r, self.seq[r["machine_id"]]) for r in self.q}

    def test_probabilities_valid(self):
        for m in (self.base, self.dirm):
            for p in self._preds(m).values():
                self.assertEqual(len(p), 4)
                self.assertTrue(all(x >= 0 and math.isfinite(x) for x in p))
                self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_default_has_no_learned_direction(self):
        self.assertEqual(self.base.dir_prior, {})
        self.assertEqual(self.base.const_prior, {})

    def test_direction_pinned_at_half_reproduces_default(self):
        pinned = _model(learn_direction=True, direction_strength=1e12).fit(self.q, self.seq)
        a, b = self._preds(self.base), self._preds(pinned)
        worst = max(abs(x - y) for k in a for x, y in zip(a[k], b[k]))
        self.assertLess(worst, 1e-9)

    def test_learned_direction_in_bounds(self):
        for v in list(self.dirm.dir_prior.values()) + list(self.dirm.const_prior.values()):
            self.assertTrue(.05 <= v <= .95)

    def test_joint_em_loglik_monotone_default(self):
        ll = [x["loglik"] for x in self.base.joint_trace]
        self.assertTrue(all(b >= a - 1e-6 for a, b in zip(ll, ll[1:])), ll)

    def test_deterministic_refit(self):
        again = _model(learn_direction=True).fit(self.q, self.seq)
        self.assertEqual(self._preds(again), self._preds(self.dirm))


if __name__ == "__main__":
    unittest.main()
