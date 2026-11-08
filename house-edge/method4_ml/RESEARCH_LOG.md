# Method 4 research log (Person D)

Metric: per-query squared error to the simulator's true 5-class probabilities, averaged per episode, per condition, divided
by the frozen reference per condition; points = 60 / (1 + mean ratio). Validation = 96 episodes (48 tuning, 48 audit, kit
split). Rule: select on tuning only; check audit once per final candidate; adopt only if both halves move the same way.
All validation-target reads below are in analysis/ scripts (diagnosis), never in fit/predict.

## Current strongest model
v8 = v7 + learned exact 2-vector size menu (`size_menu.py`, #35): tuning 58.309 / audit 58.287 / full 58.297.

v7 = v6 + 50/50 structured offset-logistic correction: tuning 58.260 / audit 58.242 / full 58.249 (keep-check vs v6 KEEP, P .986; vs team FINAL +0.104 tuning / +0.256 audit: below the +0.15 tuning bar of rule 6).
Team comparison (adi round 7, stdlib only, 58.214 full): v7 is +0.035 full; the gap is within what 6 vs ~17 correction terms and
the tree blend can explain, so treat them as equivalent in strength.

v6 (previous) = v5 + bagging over 5 tick samples: tuning 58.245 / audit 58.153 / full 58.199. Honest expected gain over a typical single
tick sample: +0.033 full (v5's 58.190 was partly sampling luck; mean single sample 58.166).

## Previous baseline (immutable reference for the research round)
v5 (`model.py`, commit a0e40e9): tuning 58.236 / audit 58.143 / full 58.190 (kit checker 58.189711).
Keep-check vs team FINAL (58.156 / 57.986 / 58.074): bootstrap P(better) .987, 4/4 conditions, 241 s -> keepcheck.py verdict KEEP.
NOTE: keepcheck.py is a bootstrap criterion; AGENTS rule 6 (+0.15 tuning AND an audit gain) is NOT met by v5 against the current FINAL
(+0.080 tuning). Same for v6/v7 (+0.089 / +0.104 tuning). Adoption needs a team decision to change the rule.

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

| 18 | Raw history holds signal v5 ignores (a sequence model could find it) (C/I) | cross-fitted GBM on raw lags 1-60, counts, neighbours -> v5 errors, 5 fold seeds (`analysis/raw_info.py`) | WIN: none (R^2 -0.11, all seeds < 0): sequence models ruled out for P(win). SIZE: R^2 +0.068 (+.04 to +.10), carried by whole-history payout counts; size model is calibrated except jackpot slope .90 (`analysis/size_shrink.py`); estimated value <= +0.04, not pursued. |
| 19 | Size evidence should update the win structure (short logs) | read Method 2 experiments 9-10 | ALREADY FALSIFIED by Method 2 (tuning +.097, audit -.015, train CV flat). Not repeated. |
| 20 | Stacker variance from WHICH ticks are sampled (D: variance) | 5 independent tick samples; bag of 3 / 5 (`analysis/bagging.py`) | SELECTION BIAS FOUND: v5's sample 0 was the luckiest of 5 (full 58.119-58.190, mean 58.166). Bag of 5: 58.245 / 58.153 / full 58.199, above every single sample on both halves. ADOPTED as v6 (`bags=5`). |

| 21 | A small STRUCTURED correction with the gate logit as a fixed offset generalizes better than trees (adi round 7 idea, extended) (H: objective/inductive bias) | offset-logistic (IRLS) on size-state, change-posterior and signal terms; l2 and blend grids (`logit_stack.py`, `analysis/logit_stack_test.py`, `logit_stack_grid.py`) | ADOPTED as v7: adi's 6-term form reproduces in my pipeline (58.211 full vs their 58.214); `d + chg` (+ P(change) x logit) and a 50/50 blend with the trees give 58.260 / 58.242 / full 58.249. Surface is flat in l2 and spec (58.24-58.25). Keep-check vs v6: P(better) .986, 4/4 conditions. |
| 22 | Machines of a shared_change episode flip at the SAME tick, so pooling tau across the 8 machines sharpens each one (L: structure) | train: within-episode sd of posterior flip time (`analysis/tau_sharing.py`) | STRUCTURE CONFIRMED (0.9 ticks within shared episodes vs 18.6 between; drift 12.7) but GAIN FALSIFIED: the mixture model (`shared_flip.py`) moves shared-change error .0245 -> .0243 (the flip time is already pinned by each machine's own post-flip ticks) and HURTS short logs (.0612 -> .0664: false 'shared' calls on sparse logs). Not used. |
| 23 | The episode's condition (stable / drift / shared) is identifiable from histories alone (B: regimes) | P(H_shared) per condition, no labels | CONFIRMED as a finding: mean P(H_shared) = 1.000 (shared), 0.002 (drift), 0.000 (stable) on validation. NOT exploitable here: each machine's own posterior already says whether and when it flipped (only 7 of 768 validation machines are ambiguous). Possibly useful if round 2 changes the flip structure. |
| 24 | RETRACTION | first run of #21 used hand-counted column indices that were off by 3 | Every term was built from the wrong columns (adi's correction scored BELOW the gate alone, contradicting adi's 58.214); `check_layout()` now asserts the layout, results above are from the fixed run. |

| 25 | Product-of-marginals bias: win rate and size share depend on the SAME hidden size state, so E[pw * s_i] != E[pw] E[s_i] (objective-aligned decomposition, mine) | regress class residuals on the covariance term (pw dev x s_i dev) per type; in-sample upper bound (`analysis/covariance_test.py`) | FALSIFIED as a score source: slopes ~0 for 1x/2x/5x (R^2 <= .05), -0.4..-0.6 only for the tiny 20x class; optimistic in-sample bound 58.249 -> 58.263 (+0.014). |
| 26 | Round-2 risk: a stacker tuned on round 1 may not generalize to a new mechanism (J/F: distribution-aware) | `round2.py`: one fit, four variants, choose on the new round's TUNING half with a margin over 'gate only' | BUILT + rehearsed on main data (picks v7; 47 s cached). Guards against a stacker that stops helping; prints learned structures with flags. |

| 27 | Ensembling independently built models decorrelates errors (D) | average full 5-class vectors of v7, adi.sizewin, method2 FINAL; weights chosen on tuning (`analysis/ensemble_test.py`) | FALSIFIED: best (2,1,0) tuning 58.262 / audit 58.247 / full 58.253 vs v7 58.260 / 58.242 / 58.249 (+0.004). Equal weights worse on audit (58.194). The models share one gate, so their errors are too correlated. |

| 28 | Size-state offset-logistic correction of P(1x\|win) from whole-history payout counts (`analysis/size_logit_test.py`) | fixed size-model logit as offset; typed whole-history and last-20 share terms; l2 1-1000; 51k examples from all histories | FALSIFIED: full 58.249 -> 58.238 .. 58.248 (-0.002 to -0.012); coefficients ~0 (the size model already uses the counts). |
| 29 | Hidden simulator structure in histories (`analysis/hidden_hunt.py`; histories only, 3072 machines, 686k ticks) | tick mod k (k=2..48), win autocorr lags 13..80, cross-machine corr lags 0..12, big-win hazard vs gap, win rate vs absolute tick, machine index vs type | NOTHING NEW: mod-k max |z| 4.3 (chance for ~1100 cells); autocorr at 14/21/28 is the geometric echo of the lag-7 structure (.158 -> .073 -> .032 -> .016); cross-machine corr .007-.014 (R^2 < .0002); big-win hazard decay is the known on/off + size state (win_category / raw_info already cover it); no position or machine-index effects. |

| 30 | A high-power proxy (win Brier on 111k held-out HISTORY ticks, train-only fit; `analysis/power_harness.py`, `power_grid.py`) can select blends / corrections better than the 48-episode halves | paired outcome-Brier on val+test histories, grid of tree blend x lr weight x lr spec | PROXY INVALID. It preferred raw trees (blend 1.0) over the .4 gate mix by ~0.54e-3 Brier (~+0.09 pts if it held), but on real validation blend 1.0 scores 58.146 and blend 0.7 scores 58.230 (tuning) vs 58.245 (v6, .4) / 58.260 (v7). Cause (hypothesis): random history ticks do not have the query regime (changes sit 40-100 ticks before the cutoff, #5), so the proxy mis-ranks stacker weights. Nothing adopted; do not use the proxy for stacker/blend selection. |

| 31 | Covariate shift: stacker/correction train on random history ticks but queries sit at the cutoff (idea from #30); reweight training rows by the query/train density ratio (discriminator on features, cross-fitted, no labels; `cshift`, `_shift_weights`) | weights ~ odds^cshift, clip 10, applied to the HGB stackers and the offset-logistic | NO GAIN: cshift=1 tuning 58.261 vs 58.260. The shift is real (discriminator AUC .97; mean P(change) .25 on training ticks vs .51 at queries; nobs 107 vs 184) but the gate posterior is already calibrated at queries, so reweighting changes nothing. Option kept, default off. |

| 32 | More bagging lowers stacker variance further (10 vs 5 tick samples) | `bags=10` | NO GAIN: tuning 58.259 vs 58.260; bagging is saturated at 5 (and doubles cold feature time). |

| 33 | The win-gate family (singles, XOR pairs, AND pairs; lags 1-12) misses real structures: XOR triples (method3's open question), OR, AND-NOT, or a free 4-cell pair partition. GBM residual tests (#18) cannot see parities, so this needed a direct test | 192 stable TRAIN machines: choose the best structure per family on the first half by Beta(1,1) marginal likelihood, score the second half (`analysis/triple_holdout.py`) | FALSIFIED, family complete: an alternative beats the gate family (2-nat margin + family-size penalty) on 0-3 of 192 machines; held-out Brier vs gate family +0.00017 (OR), +0.00019 (AND-NOT), -0.00003 (4-cell), 0 (triples). Best-triple-only Brier .146 vs .092: method3's triple picks were in-sample selection over 220 candidates. |

| 34 | Generator constraints (team brainstorm: discrete rate menu / signal as a measurement / exact type constraints), probed on validation TRUTH (`analysis/generator_probe.py`, diagnosis only) | distinct true values; spread by type x signal x on/off; distinct size vectors | Win rates CONTINUOUS (766 distinct of 768, no simple fractions): no discrete menu. Signal: mean shift per step smaller than within-level sd (3 values only): already modelled. SIZES: exactly 2 vectors per type, IDENTICAL across all machines -> exploited in #35. |
| 35 | v7's size predictions drift off the exact 2-vector segment; snapping onto a menu learned from HISTORY ticks removes that error (`size_menu.py`; bound `analysis/size_menu_test.py`, learned `analysis/size_menu_learn.py`) | onehot(history payout at t) ~ lambda_t V1 + (1-lambda_t) V2, lambda_t from the size model on seq[:t]; alternate; menu_k ticks per machine (8/16/32/64 -> tuning 58.292/58.300/58.309/58.311) | ADOPTED as v8 (menu_k=32): tuning 58.309 / audit 58.287 / full 58.297 vs v7 58.260 / 58.242 / 58.249 (+0.049 / +0.045 / +0.048; both halves). Bound with the TRUE vectors 58.303 full (learned V already reaches it). Leakage checks 1-4 pass (new check 4: menu inputs stop strictly before their history label tick). Below the +0.15 rule-6 bar, like v5-v7. |

| 36 | Round-2 safety: the size snap is an assumption (exactly 2 shared vectors per type) that round 2 may break; a no-target self-check can tell (`size_menu.check_menu`, `menu_check=True`) | 2-fold cross-fit BY MACHINE on the menu's own history rows: learn V on one fold, paired size-Brier raw vs snapped on the other; keep a type's snap only if gain >= 1 SE. Controls on synthetic data with known truth | WORKS: positive control (true 2-vector menu) t=+5.6 KEEP; negative control (per-machine random sizes) t=-14.9 OFF. Main data: steady t 2.4, balanced 3.0, jackpot 3.0, all KEEP -> predictions identical to v8 (58.309 / 58.287 / 58.297). round2.py prints the self-check and a menu-off diagnostic (58.260 / 58.249 = v7 exactly). Runtime: cold fit + test at 2 workers / 2 BLAS threads 1104 s (18.4 min; partly contended), CSV byte-identical to the 8-worker build (3efae9a5). |

| 37 | Size-state position lambda is miscalibrated (validation bins showed truth > prediction in every middle bin) | logistic recalibration of lambda per type (+ short-log terms) by likelihood of HISTORY payouts, 82k rows incl. short-masked copies (`analysis/lambda_calib.py`); room measured in `analysis/size_state_room.py` (state oracle 58.654) | FALSIFIED: slopes ~1 on history ticks; 58.297 -> 58.300 full (noise). The bin pattern was sampling noise (42-78 queries per bin). Short-log state room is real but method2 exps 8/12/13 show it does not survive the audit half. |
| 38 | Round-2 widening (adi.r2's gate: XOR triples + size split at both lags) costs nothing on main | `gate = GATE + {xor_order: 3, size_split: both}` | tuning 58.318 vs 58.309 (no regression). Kept as the round-2 gate in the candidate below. |
| 39 | adi.cook fixes short logs (two-level size-split cells, kappa by train evidence) where v8 is weakest -> pooled with v8 they decorrelate (unlike #27, cook changes the gate on short logs) | `method4_ml.cookmix`: 50/50 linear pool v8 + adi.cook, w FIXED in advance (not tuned), v8 size menu re-applied to pooled sizes; cook's gate patch removed before every v8 call | cookmix (v8 gate): tuning 58.323. With the widened gate: tuning 58.333 / audit 58.299 / full 58.314 vs v8 58.309 / 58.287 / 58.297 (+0.024 / +0.012 / +0.017; both halves; short .0528 -> .0523 tuning). REJECTED ON RUNTIME: cold 2-core times (2 workers, 2 BLAS threads): v8 1104 s, widened-gate v8 with bags=2 1516 s, adi.cook 1188 s -> v9 >= 45 min, cookmix (v8 gate) ~38 min, both over the 30-min organizer limit; the full v9 run was stopped at 40 min. The widened gate alone (+0.009 tuning) leaves < 5 min margin. v8 stays the entry. Code kept (`cookmix/`) for a faster runner or a cheaper cook. |
| 40 | v9 robustness: types, short-log shape and short threshold were main-round constants (crash / silent mis-masking if round 2 changes them) | learn them in fit from the round's queries + histories (`learn_short_shape`, `type_order`); new types get new indices, other shapes use harvested real masks | main: identical to v8 (58.309 / 58.287; learned classic 40 + 24). Stress kits: a 4th type and a 'last 30 + every 4th' mask run without errors. |
| 41 | The gate family sees only OWN win bits; cross-machine / size-category / own-XOR-neighbour rules are invisible to it and trees cannot find parities (`polarity.py`: per-machine evidence bank on its own history, best / 2nd-best z and signed shift, 12 features; optional 'pol' logistic term) | main data (must not hurt) + synthetic stress kits with known truth (`adapt/stress/make_kit.py`) | main: 58.305 / 58.287 (= v8, cost +1% runtime). CROSS-machine kit (half the machines driven by a neighbour lag): v8 34.95 full -> v9 trees 37.72 -> v9 + pol logistic 39.84 (+4.9; both halves). Insurance for round 2, free on main. Leakage check 1b passes. |
| 42 | E1: cook's gain is short-log specific, so a 2-weight (short / long) pool of v8 + adi.cook beats the fixed 50/50 (`analysis/gated_pool.py`, 2-fold cross-fit on the kit halves, paired episode bootstrap) | out-of-fold contest points | FALSIFIED: gated 58.306, global w 58.308, fixed .5 58.307 vs v8 58.297; P(better than v8) .71 / .79 / .71 (< .9), P(gated > fixed) .46; chosen weights flip between folds (w_short .2 vs .6). Cook alone 1397 s: not shippable either. Champion stands. |
| 43 | E2: v8-vs-cook disagreement in P(win) flags where pooling helps | Spearman on validation queries | FALSIFIED as a gate: disagreement tracks v8's error (rho .42) but not the pooling gain (rho .02): it marks uncertainty, not which model is right. |

Eliminated classes (also: factorization bias, shared-tau pooling, raw-history sequence models): change-model shape (4, 5, 6, 7), cross-machine information (10, 11), richer population priors (8, 9),
more learner capacity / data / objective tweaks (3, 12, 13, 15).

## Remaining unexplained error and highest-value next hypothesis
- Remaining: per-machine cell-rate uncertainty (jackpot on-cells), and short_history (40 consecutive + 24 isolated ticks).
- Bounded (#17): marginalizing short logs' missing gate inputs, estimated +0.03.
- Conclusion this round: no new mechanism beyond v5 survived. 17 hypotheses tested; the error that remains behaves like
  irreducible per-machine parameter uncertainty given the data (calibrated, unlearnable residuals, flat in data/capacity).
- Best use of remaining effort: round 2 (a refit on new mechanisms), not more main-round search.
- Round 2 changes "which temporal relationships recur": v5 learns everything in fit() (structures, cell priors, signal betas,
  size model, stacker), so nothing above is hard-coded.

## Round-2 procedure (method4)
1. `python -m method4_ml.round2 --data data/adaptation --out build/round2 --no-cache` (about 15 min on 2 cores, ~5 on 8).
2. Read FLAGS and the variant table. The chosen variant is written to `build/round2/chosen.json`.
3. Package: `python -m method4_ml.make_entry --data data/adaptation --out build/adapt_m4 --kwargs "$(cat build/round2/chosen.json)"`
   (chosen.json holds blend / lr_weight; the ZIP refits on the runner, so check its CSV equals build/round2/submission.csv).
4. If scikit-learn is not allowed on the runner, ship adi.sizewin (stdlib only, 58.214 full) or the numpy-free parts of this model.
