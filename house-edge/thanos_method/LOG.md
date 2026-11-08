# Clean-slate challenger log

## 1. Pre-registered evaluation of EGP (DESIGN.md), 2026-11-08
Leakage tests: ALL 8 PASS (`tests_leakage.py`).
Train-only 4-fold episode CV (surrogate, 3072 outcome rows at cutoffs c..c-3): 5-class Brier out-of-fold .35865 vs in-sample hyperparameters .35850
(gap +.00015, 95% CI [-.00007, +.00039]): the hyperparameter fit does not overfit. Pooled per-type frequency .57105.

Learned from TRAIN histories alone: pairs (1,3)/(2,5)/(4,7) with size lag = smaller lag (screen mass .79 on those three; the tail is wrong-size-lag
variants); P(win-shift direction = size direction) ~ .9 (NEW: Momin averages the direction 50/50); gamma .47 / .53 / .29; signal slopes .32-.39;
P(no flip | long log) .329 (truth 1/3 stable); flip position mode .65-.70 of the cutoff.

Official metric, full validation:

| protocol | champion | EGP | change | P(gain) | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inductive | 58.332 | 58.222 | -0.110 | .36 | -7.4% | +28.3% | -13.5% | -17.7% |
| transductive | 58.355 | 58.267 | -0.089 | .39 | -7.3% | +25.6% | -11.9% | -20.1% |

(ratio changes: negative = EGP better). Pre-registered blend .5/.5 (inductive): 58.379 (+0.047, P .69, top 5 episodes = 147% of the gain) -> REJECTED.
Runtime (fit + test, 2 threads, 1 process): inductive 225 s, transductive 372 s; peak RSS 131 / 188 MB. Champion: 1632 s.
VERDICT: EGP NOT PROMOTED. Champion retained.

## 2. Post-hoc hypothesis (formed AFTER seeing section 1, so validation cannot confirm it)
H-route: the sequence likelihood (EGP) is better wherever the log is complete; the champion is better on sparse (short) logs, where EGP discards
the 24 isolated ticks and the first 8 ticks of the 40-tick block. Routed model = EGP on long logs, champion on short logs (log shape is
observable from the mask; no condition labels). Validation arithmetic would give ~58.47 transductive, but that number is CONTAMINATED by selection.

Independent test (pre-registered here, before running): TRAIN queries, outcome at the real cutoff (train label) and at pseudo-cutoffs c-1..c-15
(history truncated; outcome = observed history tick). Both models fitted transductively (champion = saved checkpoint, EGP = egp_trans).
Caveat that favours the CHAMPION: its stacker trained on random train history ticks >= 100 as labels, so some pseudo-cutoff outcomes are in its
training set; EGP's 16 global parameters used them only through the likelihood.
Decision rule: H-route survives only if, on LONG train logs, EGP - champion paired 5-class Brier < 0 with episode-bootstrap P >= .95.
If it survives, the routed model becomes a candidate (it still needs runtime: champion + EGP in one entry).

## 3. Breakthrough brief, Phase 1: evidence map (code read, not only summaries)
Code facts (champion = mominsreverseengineering method4_ml/model.py + adi/gate.py + shared_base.py):
- Population priors (cell Betas, signal shifts, flip prior, structure prior, Momin's shared base / gamma) are fitted by MARGINAL / SEQUENCE likelihood
  over whole histories already. Only the tree stacker, the offset-logistic correction and the size menu use pseudo-examples: K=6 random ticks in
  [100, cutoff) per episode + K_short=6 copies masked as short logs (40 consecutive + 24 uniform older), 5 bags, from train + val + test histories.
- Short logs: adi's gate uses a tick when ITS OWN lags are observed; isolated older ticks are dropped; p_change_short learned by evidence.

| failed experiment | assumption changed | why it failed | independent? | failure type | narrower untested hypothesis |
| --- | --- | --- | --- | --- | --- |
| #31 covariate-shift reweighting | stacker rows reweighted to the query feature density | shift REAL (AUC .97, P(change) .25 vs .51, nobs 107 vs 184), gain 0 | yes | bias absent: gate posterior already calibrated at queries | none: the reweighting answers Phase 2's central question for the stacker |
| #30 history-tick proxy | random history ticks represent queries | mis-ranked blends (regime differs at cutoff) | yes | proxy bias | near-cutoff pseudo-queries (c-1..c-15) were never used as a proxy -> used in section 2 |
| K 6/12/24, bags 5/10 (#32, v5) | more overlapping pseudo-examples | flat | yes | not data-limited | Phase 2C: effective-sample-size is irrelevant to a stacker that is already flat in K |
| method2 exp 12-13 (missing-tick filter) | short logs: integrate the hidden lag bits of isolated ticks | tuning +.09 / +.12, audit -.01 / -.04 | add-on to the gate, then stacked | variance (24 short episodes per half) | the same marginalization inside a model WITHOUT a stacker (EGP) never tested |
| #22 shared tau | pool flip times in shared episodes | own post-flip ticks pin tau | yes | no information left | none |
| #45 shrink / temperature | over / under-personalisation | train outcomes contradict validation lead | yes | selection noise | none |
| #12, #18, #28 direct / sequence learners | learn residual / sizes from raw history | R^2 < 0 | yes | nothing to learn | none |
| method2 exp 9-10, #19 size -> win | size evidence updates structure | audit fails | partly (same gate) | variance | in EGP it is native (joint likelihood), no add-on |

Phase 2 Experiment A (no targets): champion short-mask augmentation vs real short queries (768 real short logs, train + val + test):
block 40 vs 40(+label tick), older ticks 23.84 vs 23.84, older-tick relative position quantiles (.10/.25/.50/.75/.90) .095/.243/.490/.738/.888 vs
.097/.246/.494/.744/.890, median gap 5 vs 5, adjacent pairs 5.68 vs 5.61. NO MASK MISMATCH. The position mismatch (#31) is real but was shown harmless.
Experiment C: the only pseudo-example consumers are flat in K and bags; the sequence-likelihood model (EGP) has a hyperparameter generalization gap of
+.00015 Brier (CI includes 0). Over-counting of overlapping examples is not a recoverable error source.
Phase 4: EGP IS the "correctly specified causal sequence likelihood with shared population parameters" (option C of Phase 4). Section 1 measures it:
better than the champion on complete logs (-7% to -20% error), worse on short logs (+26-28%).

## 4. Post-hoc variant E2: per-structure tick sets (`tickset="own"`), formed AFTER section 1
Cause of EGP's short-log loss (hypothesis): the pre-registered "common tick set" needs all 8 lags observed, so a short log's 40-tick block gives
32-34 usable ticks; each structure needs only its own lags (38-40 usable). E2 lets each structure use its own ticks and gives the ticks a
structure cannot use a plug-in marginal term (machine win rate + size frequencies) so evidences stay comparable. Complete logs: identical
predictions (checked: max diff 0). Validation is contaminated for this choice; the deciding evidence is train-only.
Decision rule (written before running): E2 replaces EGP only if train 4-fold episode CV shows short_history out-of-fold Brier lower than EGP's
with paired episode bootstrap P >= .9, and no other condition worse (long logs are identical by construction).

## 5. Results of the independent train test (section 2) and E2 on validation
Section-2 test (TRAIN outcomes at c..c-15, both transductive, EGP = original common tick set):
LONG logs 9216 rows / 72 episodes: EGP - champion Brier -0.001032, 95% CI [-0.001715, -0.000313], P(EGP better) .997 (real cutoff only -0.00161, n 576).
SHORT logs 3072 rows / 24 episodes: +0.000611 [-0.001272, +0.002572], P .27. -> H-route's long-log claim PASSES its pre-registered rule.
E2 (tickset own) inductive, full validation (CONTAMINATED: chosen after section 1): 58.567 vs champion inductive 58.332 (+0.235, CI [+0.018, +0.429],
P .982, 62/96 improved; tuning 58.424 vs 58.315, audit 58.732 vs 58.355; stable -8.0%, short -15.5%, drift -13.1%, shared -18.3%;
top-5 share 69%, without top 5 +0.073, without top 10 +0.009). Deciding evidence still pending: CV (section 4 rule) and the section-2 test rerun with E2.
