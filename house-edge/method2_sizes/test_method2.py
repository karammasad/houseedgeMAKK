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


class TestMissFilter(unittest.TestCase):
    """Experiment 12: exact marginalization over unobserved gate inputs (no data files needed)."""
    def setUp(self):
        import random
        from method2_sizes.missfilter import MissFilter
        self.mf = MissFilter({"balanced": [.8, .07, 50, 50, .3, .3]}, G=6)
        rng = random.Random(1); self.s = [rng.choice((0, 1, 1, 2)) for _ in range(30)]
        for t in (6, 7, 13, 14, 15, 22):
            self.s[t] = -1
        self.r = {"machine_type": "balanced", "signal": 0}

    def test_matches_brute_force_enumeration_of_hidden_bits(self):
        mf, w = self.mf, [-1 if v < 0 else int(v > 0) for v in self.s]
        obs = [x for x in w if x >= 0]; p = min(.95, max(.05, sum(obs) / len(obs))); pair, o = (2, 5), 1
        pr_on, pr_off = mf._grid_prior("balanced", 0); z_fast, _ = mf._evidence(w, pair, o, pr_on, pr_off, True, p)
        a, b = pair; tot = []
        for i in range(mf.G):
            for j in range(mf.G):
                ll = pr_on[i] + pr_off[j]
                for t in range(b, len(w)):
                    if w[t] < 0:
                        continue
                    va, vb = w[t - a], w[t - b]; rate = 0.0
                    for ha in ((va,) if va >= 0 else (0, 1)):
                        for hb in ((vb,) if vb >= 0 else (0, 1)):
                            pa = 1.0 if va >= 0 else (p if ha else 1 - p); pb = 1.0 if vb >= 0 else (p if hb else 1 - p)
                            rate += pa * pb * (mf.g[i] if (ha ^ hb) == o else mf.g[j])
                    ll += math.log(rate if w[t] else 1 - rate)
                tot.append(ll)
        m = max(tot); self.assertAlmostEqual(z_fast, m + math.log(sum(math.exp(x - m) for x in tot)), places=9)

    def test_no_unknown_ticks_means_identical(self):
        full = [max(0, v) for v in self.s]   # fully observed: nothing to marginalize
        mf = self.mf; mf.none_mean = {"balanced": .4}; K = 57; mf.prior = [1 / K] * K
        self.assertAlmostEqual(mf.pwin(self.r, full, True), mf.pwin(self.r, full, False), places=12)

    def test_valid_probability_and_missing_never_loss(self):
        mf = self.mf; mf.none_mean = {"balanced": .4}; mf.prior = [1 / 57] * 57
        p = mf.pwin(self.r, self.s, True); self.assertTrue(0 < p < 1)
        allmiss = [v if t >= 22 else -1 for t, v in enumerate(self.s)]   # sparse history: a gap is never filled with zeros
        self.assertTrue(0 < mf.pwin(self.r, allmiss, True) < 1)


class TestJointFilter(unittest.TestCase):
    """Experiment 13: joint hidden-category marginalization equals independent brute-force enumeration."""
    def test_matches_brute_force(self):
        import itertools, random
        from method2_sizes.missfilter import MissFilter
        from method2_sizes.jointfilter import JointFilter, cat
        par = dict(wST=[(2, 5)], cond=[[0, 0, 1.0] + [0] * 6, [0] * 9], part_prior=[.96, .02, .02],
                   levels={"balanced": [.2, .5]}, split={"balanced": [.7, .26, .04]})
        mf = MissFilter({"balanced": [.8, .07, 50, 50, .3, .3]}, G=4); mf.none_mean = {"balanced": .4}
        mf.prior = [1.0 if n == ((2, 5), 1) else 0.0 for n in [(p, o) for p in __import__("method2_sizes.missfilter", fromlist=["PAIRS"]).PAIRS for o in (0, 1)]] + [0.0]
        jf = JointFilter(mf, par, G=4, prune=0.5)
        rng = random.Random(4); s = [rng.choice((0, 0, 1, 1, 2, 5)) for _ in range(34)]
        for t in (8, 9, 15, 16, 17, 24):
            s[t] = -1
        r = {"machine_type": "balanced", "signal": 0}; pair, k, o, d = (2, 5), jf.klag((2, 5)), 1, 0
        ll, (U, p, s1) = jf.evidence(r, s, pair, k, o, d, True)
        lv = par["levels"]["balanced"]; qc = (1 - p, p * s1, p * (1 - s1)); a, b = pair
        pr_on, pr_off = mf._grid_prior("balanced", 0)
        for i in range(4):
            for j in range(4):
                tot = pr_on[i] + pr_off[j]
                for t in range(max(a, b, k), len(s)):
                    if s[t] < 0:
                        continue
                    lags = sorted({a, b, k}); hid = [l for l in lags if s[t - l] < 0]; f = 0.0
                    for vals in itertools.product((0, 1, 2), repeat=len(hid)):
                        c = {l: cat(s[t - l]) for l in lags if s[t - l] >= 0}; pr = 1.0
                        for l, v in zip(hid, vals):
                            c[l] = v; pr *= qc[v]
                        rate = jf.g[i] if (int(c[a] > 0) ^ int(c[b] > 0)) == o else jf.g[j]; lk = rate if s[t] > 0 else 1 - rate
                        if s[t] > 0:
                            st = d if c[k] in jf.grp else 1 - d; lk *= lv[st] if s[t] == 1 else 1 - lv[st]
                        f += pr * lk
                    tot += math.log(f)
                self.assertAlmostEqual(tot, ll[i * 4 + j], places=9)
        v = jf.run(r, s, True)
        self.assertAlmostEqual(sum(v), 1.0, places=9); self.assertTrue(all(x >= 0 for x in v))


if __name__ == "__main__":
    unittest.main()
