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

    def test_twolag_mixture_valid_and_bounded(self):
        from method2_sizes.twolag import TwoLagSizes, MAPS, PAIRS
        self.assertEqual((len(PAIRS), len(MAPS)), (28, 10))
        tl = TwoLagSizes(_model(), mode="mix", pairs="uniform").fit(self.q, self.seq)
        self.assertTrue(.01 <= tl.v <= .99)
        for r in self.q:
            p = tl.predict(r, self.seq[r["machine_id"]])
            self.assertTrue(all(x >= 0 for x in p)); self.assertAlmostEqual(sum(p), 1.0, places=12)
            w = tl.weight_new(r, self.seq[r["machine_id"]]); self.assertTrue(0 <= w <= 1)

    def test_deterministic_refit(self):
        again = _model(learn_direction=True).fit(self.q, self.seq)
        self.assertEqual(self._preds(again), self._preds(self.dirm))


class TestJointMath(unittest.TestCase):
    """Experiment 9: size -> win feedback math on synthetic inputs (no data or model fit needed)."""
    class _Sizes:
        def predict(self, r, s):
            return [.4, .3, .2, .1]

    def _P(self, logE):
        # two structures; structure 0 says P(win)=.9, structure 1 says .1; equal win evidence
        flat = [(math.log(.5), .9, 0), (math.log(.5), .1, 1)]
        return dict(flat=flat, z0=0.0, logE=logE, post_s=[.5, .5], tabs=None, K=1)

    def test_no_feedback_is_win_posterior_mean(self):
        from method2_sizes.joint import assemble
        p, post = assemble(self._Sizes(), None, {}, [], self._P([0.0, 0.0]), feedback=False, size_from_adi=False)
        self.assertAlmostEqual(1 - p[0], .5, places=12); self.assertEqual(post, [.5, .5])

    def test_equal_size_evidence_changes_nothing(self):
        from method2_sizes.joint import assemble
        p, _ = assemble(self._Sizes(), None, {}, [], self._P([-3.0, -3.0]), feedback=True, size_from_adi=False)
        self.assertAlmostEqual(1 - p[0], .5, places=12)

    def test_feedback_moves_toward_size_supported_structure(self):
        from method2_sizes.joint import assemble
        p, post = assemble(self._Sizes(), None, {}, [], self._P([0.0, math.log(1 / 9)]), feedback=True,
                           size_from_adi=False)
        self.assertAlmostEqual(post[0], .9, places=12); self.assertAlmostEqual(1 - p[0], .9 * .9 + .1 * .1, places=12)
        self.assertAlmostEqual(sum(p), 1.0, places=12); self.assertTrue(all(x >= 0 for x in p))


if __name__ == "__main__":
    unittest.main()
