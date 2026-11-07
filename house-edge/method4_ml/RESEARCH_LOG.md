# Method 4 research log (Person D)

Metric: per-query squared error to the simulator's true 5-class probabilities, averaged per episode, per condition, divided
by the frozen reference per condition; points = 60 / (1 + mean ratio). Validation = 96 episodes (48 tuning, 48 audit, kit
split). Rule: select on tuning only; check audit once per final candidate; adopt only if both halves move the same way.
All validation-target reads below are in analysis/ scripts (diagnosis), never in fit/predict.

## Current strongest model (immutable baseline for this round)
v5 (`model.py`, commit a0e40e9): tuning 58.236 / audit 58.143 / full 58.190 (kit checker 58.189711).
Keep-check vs team FINAL (58.156 / 57.986 / 58.074): bootstrap P(better) .987, 4/4 conditions, 241 s -> KEEP.

## Phase 1: diagnosis (observed evidence)
| question | measurement | result |
| --- | --- | --- |
| Where is the error? | oracle swap, full validation (`analysis/diag_v5.py`) | true P(win) -> 59.57 (+1.38); true sizes -> 58.59 (+0.40). Win side dominates; short_history is worst on both sides (ratio .058). |
| Systematic residual? | episode-grouped 5-fold CV GBM on v5's own features -> v5's errors | R^2 = -0.033 (P(win) error), -0.039 (P(1x\|win) error). Nothing the model sees predicts its errors. |
| Calibration | slope of truth on gate P(win), by condition x type (`analysis/calib.py`) | slope .95-1.05, bias within +-.02 in all 12 slices. |
| Underfit / capacity? | 72-config HGB grid; K = 6/12/24 training ticks per episode | flat (58.17-58.24); K=6 == K=24. Not data- or capacity-limited. |
| Distribution shift? | split audit (`analysis/anomalies.py`) | train / val / test match on type, signal, payout mix, cutoffs, short share (25%); no duplicates, no leakage. |
| Stability | tuning vs audit; bootstrap vs FINAL | same direction on both halves; P(better) .987. |
Interpretation (hypothesis, well supported): the remaining error is mostly per-machine PARAMETER uncertainty (each machine's own
cell win rates from ~10-25 observations; worst in jackpot on-cells), not a missing learnable function of the features.
RETRACTED: an approximate posterior-variance check suggested drift/shared were 3x overconfident; direct calibration shows they
are not (their MSE equals stable's). The approximation understated variance for machines with a change.

## Phase 2-7: hypotheses, experiments, outcomes
Each: mechanism -> evidence -> falsifier -> result. Gains are vs the stated baseline on the tuning half unless noted.

| # | hypothesis (direction) | smallest test | outcome |
| --- | --- | --- | --- |
| 1 | Win rate depends on observed `signal` within type (L: structure) | on-rate by type x signal x size state on truth; prior shift learned by train marginal likelihood (`signal_shift.py`) | KEEP. Shift +.09/step in jackpot; gate 58.080 -> 58.156; same choice on train vs all histories. (adi found +0.30 slope independently.) |
| 2 | Stacker blind to hidden size state that moves the win rate (C/D: representation) | add size model's P(1x\|win) to stacker features | KEEP. 58.175 -> 58.234; audit +; keep-check KEEP. |
| 3 | Residual correction (A) | cross-fitted residual GBM | FALSIFIED: CV R^2 < 0. |
| 4 | Change is not a pure swap: rates differ after the change (L) | train: rate z-scores pre vs post detected change (`analysis/swap_test.py`) | FALSIFIED: var(z) 1.04-1.15 vs stable control 0.90-1.16. |
| 5 | Uniform change-time prior is miscalibrated (L) | train: posterior change age; val: error vs age (`analysis/tau_position.py`) | Structure FOUND (changes 40-100 ticks before cutoff; none in stable) but FALSIFIED as error source: posterior already sharp, error flat in age. |
| 6 | Size-state mapping reverses at a change (B: regimes) | train: size-gate direction pre vs post (`analysis/size_state_flip.py`) | FALSIFIED: kept in 94-97 of ~100 machines (= control). |
| 7 | Gradual drift (ramp) between states (I) | share of mid-range true rates by condition (`analysis/drift_shape.py`) | FALSIFIED: <= 4% mid-range in drift, same as stable. |
| 8 | Per-machine latent quality shifts all cells (B/K: hierarchical) | random effect marginalized on a grid, sigma by train evidence (`machine_shift.py`) | FALSIFIED: train picks sigma 0-.15 (1-5 nats); 58.077 -> 58.079 full. |
| 9 | Population prior shape wrong (mixture-of-Betas, ebbr-style) (C) | `mixprior.py` per type x signal | FALSIFIED: 57.68-57.69 (no gain). |
| 10 | Episode sharing of structure / rates (L) | `analysis/episode_structure.py`, `episode_rates.py` | FALSIFIED: same-structure 33.1% vs 33.5% chance; rate ICC .07-.09. |
| 11 | Neighbour machines' past outcomes drive the win gate (E/L) | held-out log-lik with best (neighbour, lag) split (`analysis/neighbour_win_gate.py`) | FALSIFIED: worse than own gate (-0.328 vs -0.299). |
| 12 | Factorization [1-p, p*s] loses win x size coupling -> direct size stacker (H/D) | HGB on P(1x\|win) blended into the size model | FALSIFIED: hurts (58.236 -> 58.196 at weight .6). |
| 13 | Objective / capacity misalignment (H) | HGB grid, offset boosting from gate logit | FALSIFIED: flat. |
| 14 | Model complementarity: average gate variants (D) | adi + method 1 P(win) | FALSIFIED: worse at every weight. |
| 15 | More data for priors: learn from val/test histories (J) | pooled priors and sizes | FLAT: +0.035 (priors), 0 (sizes). |
| 16 | adi's own `cell_signal` vs my `signal_shift` (teammate code) | full pipeline, alone and combined | EQUIVALENT: gate 58.156 both; pipeline 58.233 (cell_signal) vs 58.236 (mine); combined = cell_signal alone. v5 unchanged. |
| 17 | Short logs waste their 24 isolated older ticks (J/L) | mask long val machines: last 40 / + isolated / + isolated with lag inputs revealed (`analysis/short_isolated_bound.py`) | Ceiling .00483 -> .00325 MSE, but it also reveals ~144 extra usable ticks. Marginalizing only the missing inputs recovers mixture-rate information, which the isolated ticks already give today (.00502 -> .00483, 4%). Estimated +0.03 points: NOT BUILT (cost > value before round 2). |

Eliminated classes: change-model shape (4, 5, 6, 7), cross-machine information (10, 11), richer population priors (8, 9),
more learner capacity / data / objective tweaks (3, 12, 13, 15).

## Remaining unexplained error and highest-value next hypothesis
- Remaining: per-machine cell-rate uncertainty (jackpot on-cells), and short_history (40 consecutive + 24 isolated ticks).
- Bounded (#17): marginalizing short logs' missing gate inputs, estimated +0.03.
- Conclusion this round: no new mechanism beyond v5 survived. 17 hypotheses tested; the error that remains behaves like
  irreducible per-machine parameter uncertainty given the data (calibrated, unlearnable residuals, flat in data/capacity).
- Best use of remaining effort: round 2 (a refit on new mechanisms), not more main-round search.
- Round 2 changes "which temporal relationships recur": v5 learns everything in fit() (structures, cell priors, signal betas,
  size model, stacker), so nothing above is hard-coded.
