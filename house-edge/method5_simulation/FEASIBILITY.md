# Method 5: feasibility of simulation-based inference (Phase 1 report)

Everything here lives in `method5_simulation/`. No existing method, shared file or submission was touched.
Reproduce the data checks: `python -m method5_simulation.data_checks --data data/main`.

## 1. Is there an organizer simulator or a precise specification?
**No.** The participant guide links `github.com/tamu-datathon-org/slots-rl` as a "project reference" that "may require
access" and says it is not needed to compete. Fetching it returns **HTTP 404** (private/restricted); I did not try any
authenticated workaround. The only specification in the kit is `DATA.md`, which is qualitative ("interact at delays up to
eight ticks", "win chances and payout sizes can change", "two machines of the same type can react oppositely"). Any
generator is therefore **our own reconstruction**, fitted on the same train histories we want to predict.

## 2. How faithful can a reconstruction be? (evidence from Method 2's simulator, used only as a reference)
- Posterior predictive check (20 statistics x machine type x short/long, real = validation + test histories, disjoint
  episodes from the fit): the first version flagged 55/80 statistics; three of the causes were simulator errors (wrong
  short-log flip rate, a lag-cancelling bug, no size-state link). After correction **19/80 remain flagged (|Z| > 3),
  max |Z| = 9**: real rate tails are heavier than the Beta priors; real short-log gating slightly stronger.
- Self-consistency ceiling (our inference on simulated machines with known truth vs on real ones): stable z = +0.9,
  short z = -0.8, **drift z = +1.9, shared z = +1.9** (real error ~1.6x what the generator implies).
- Conclusion: the reconstruction is **good, not exact**, and its worst gap is exactly in drift/shared change.

## 3. What could synthetic supervision buy? (the decisive argument)
An amortized predictor trained to convergence on a generator S learns S's **posterior predictive**, which is exactly
what exact Bayesian inference under S computes. The team's current models (Method 2 / Adi's gate, 58.07-58.21) already
compute that, because they are built from the same structural findings (the generator was fitted to the same data).
So the synthetic route can at best **match** the existing level (minus fidelity loss, plus the drift/shared
misspecification above) and cannot supply new real information: the generator's parameters came from the training
data, and its structure came from our own hypotheses. Its one real upside would be variance reduction (exact targets
instead of one-hot outcomes), which only matters if the learner has many parameters. **Decision: do not manufacture
a synthetic training set for the headline model.**

## 4. Is the real data trustworthy as supervision? (Phase 4 route) Yes.
- Splits disjoint (0 shared episodes/machines). Every history is exactly `cutoff` ticks (strictly below the cutoff);
  the label tick is never loaded. Short logs: 192/192 per split follow the 40 consecutive + 24 sampled layout;
  missing ticks stay `-1` (never 0).
- **521,113 self-supervised next-tick examples** with a full 8-lag context are available from histories alone
  (train 129k, validation 131k, test 261k); no validation/test labels are needed (AGENTS.md allows other splits'
  histories in `fit`).
- Loss alignment: for a prediction p, true distribution P and one-hot outcome y,
  `E||p - y||^2 = E||p - P||^2 + E[1 - sum P_k^2]`. Minimizing Brier loss on real outcomes minimizes the competition's
  squared error to the true probabilities (the second term is a constant), so real one-hot targets are unbiased
  supervision for the objective; they are only noisier than exact targets (type-mean noise floors: .38-.66).
  This is affordable only with a **low-parameter** learner: with ~500 parameters and 10^5 examples the noise averages out.

## 5. Plan: Method 5 = learned context mixture (no gate, no polarity, no size-lag, no EM)
Generic, structure-agnostic conditional statistics from a machine's own history prefix, mixed by a learned attention:
- Contexts: every single lag (8) and every lag pair (28) of the observed outcomes (win bit; payout category for sizes).
  No XOR, polarity, flip or tied-state assumption is encoded.
- Per context, for the cell the cutoff falls in: smoothed empirical win rate and count over the whole prefix and over a
  recent window, the rate in the other cells (separation), and first-half vs second-half agreement.
- A shared small network scores each context and proposes an estimate; softmax attention over contexts pools them; a
  calibration head adds machine type and signal. A parallel head predicts the size split from payout-category contexts.
- Trained offline (torch) on real self-supervised ticks with Brier loss, **episode-grouped cross-validation**, then
  exported as plain weights and run in standard-library Python. Deterministic seeds.
- Imports nothing from Methods 1-4; compared against 58.074 (Method 2) and ~58.2 (team best) with the competition metric.

Honest expectation: this will probably score below the Bayesian models (it must *learn* what they assume), perhaps
56-57.5. The scientific question is whether it recovers information they miss (the drift/shared gap), and whether it
adds diversity. A failure is informative either way.
