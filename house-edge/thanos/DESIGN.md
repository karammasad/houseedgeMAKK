# Clean-slate challenger: design, evidence review and pre-registration

Written 2026-11-08 BEFORE any code of the challenger existed or any of its results were seen.
Folder: `thanos/` (independent: imports only `common.data` for CSV loading and `common.evaluate` for the official metric).

## Phase 1: what the repository has learned

The champion (Momin's shared-base on v8, 58.355 full validation transductive, 58.332 inductive) is a stack of layers:
adi's Bayesian change-point gate (structure mixture over XOR lag sets, Beta cell priors shifted by signal, one polarity flip)
-> method2 GateSizes (size state coupled to the gate lag, joint EM) -> 3-seed x 5-bag HistGradientBoosting stacker on gate-posterior
features (blend .4) -> 50/50 offset-logistic correction -> learned 2-vector size menu snap -> Momin's "xsg" structure
(one shared base rate per gate side + a size-state shift gamma) added to the gate mixture. Feature building is 67% of its 27.2-min 2-core run.

### A. Structural evidence worth retaining (measured, most on training histories)
1. Win gate = XOR of the machine's own WIN bits at a lag pair; only three pairs occur: (1,3) .34, (2,5) .31, (4,7) .31, "no gate" ~.05 (#51, #33: triples/OR/AND-NOT/4-cell all falsified).
2. A binary size state: "payout at lag l was big (2x/5x/20x)" vs not, l = the smaller lag of the pair (method2 share_probe: 24/24, 38/38, 38/38). Loss and 1x are one category (#48 P).
3. Given a win the true size distribution is one of exactly TWO vectors per type, identical across machines (#34, #35).
4. The win rate also moves with the size state: one shift gamma per type (.46 / .53 / .30), one base rate per gate side (Momin #48; every richer tying estimated back to it).
5. Signal shifts the base win logit (~+.3 per step) inside a type (#1).
6. Changes: stable logs never flip; drift / shared logs flip exactly once, at tau/cutoff ~ .68 (+-.1); at a flip ONLY the polarity swaps (rates, size gate kept: method2 exp 11, #4, #6, #7).
7. Short logs (40 consecutive + 24 isolated ticks) rarely flip; the right flip prior differs by observable log shape (method2 exp 11: a single learned prior destroyed short logs).
8. Rates are continuous per machine (no menu, #34); the irreducible part is per-machine rate noise. Calibration is already good (#45).

### B. Architecture-specific choices that can be challenged
- Separate stages (gate -> sizes -> stacker -> correction -> snap) each fitted with its own objective; the stacker and correction exist to repair the gate's
  misspecification (Beta cells per sub-cell instead of the shared-base + gamma generator). A model that IS the generator should not need them.
- Hand-set or grid-capped pieces: long-log p_change .5 (#54), flat flip-time prior, Beta strengths capped by a grid, tau grid step 2.
- Size evidence never updates the structure / direction posterior except through method2's one-way coupling.
- 99% of compute is per-tick feature construction for the trees.

### C. Falsified hypotheses (not repeated under new names)
Raw-history sequence models for P(win) (#18: R^2 < 0), direct 5-class / size stackers (#12, #28), residual GBMs (#3), richer population priors
(mixture-of-Betas #9, machine random effect #8), neighbour / cross-machine features (#10, #11), gradual drift (#7), post-flip rate changes (#4, #6),
shared-tau pooling (#22), size->win feedback as an ADD-ON (method2 exp 9-10, #19), uniform-shrink / temperature (#45), more bags / capacity (#13, #32),
the history-tick Brier proxy for selecting stacker weights (#30).

## Phase 2-3: five candidate architectures

| | A. conditional 5-class distribution (multinomial GAM / small boosting) | B. finite-state Bayesian filter (compact HMM) | C. direct history -> probability (small TCN / GRU) | D. hierarchical state-space mixture (pooled transitions, own latent state) | E. NEW: exact generator posterior (one joint likelihood of the payout stream, exact discrete marginalization, grid-integrated rates, empirical-Bayes hyperparameters) |
| --- | --- | --- | --- | --- | --- |
| assumptions | smooth additive effects of features | Markov latent state with few states | the map is learnable from ~770 episodes | low-dim shared dynamics | the generator family of A.1-A.7 (each measured), continuous per-machine base rates |
| differs from champion | no gate at all | latent state instead of XOR cells | learned, no structure | latent dynamics instead of a flip | one model, no stages, no trees, sizes and wins inferred jointly, learned flip prior per log shape |
| information | features only (parities invisible to trees / GAMs) | order-k context would need 2^7 states x rates | raw sequences | pooled dynamics | every observed payout (win AND size) informs structure, direction, polarity, flip time |
| why it could reduce error | none shown | none: a generic HMM cannot express a lag-7 XOR without 128 states | none | uncertain | removes misspecification the stacker currently patches; Bayes-optimal under a correct family |
| overfitting risk | medium | medium (EM on many states) | high (episodes are few) | medium | LOW: ~16 continuous hyperparameters per type + a structure prior, all fitted by marginal likelihood on histories, no supervised targets |
| data needs | many labels | moderate | large | moderate | histories only (686k ticks) |
| cost | small | moderate | moderate-large | moderate | small (vectorized counts; no per-tick features) |
| falsified if | (already: #18, #12) | (already: size_hmm 2 points worse) | (already: #18) | its latent state does not beat a flip (#7: drift is abrupt) | it does not reach the gate+stack pipeline (58.25 inductive v8) on validation / train CV |

Ranking by repository evidence: E > D > B > A > C. A, B and C are contradicted by measured results (#18 shows nothing in raw
history predicts the gate's errors; trees cannot see parities; the size HMM was 2 points worse). D's distinctive element (continuous latent
dynamics) is contradicted by #7 (true rates switch abruptly). E is the only one whose every assumption is a measured property of the
generator and whose remaining freedom is estimated without labels. It is NOT "another correction to Momin": it has no gate code, no Beta cells,
no stacker, no logistic correction, no size snap and no imports from adi / method2 / mominsreverseengineering.

Selected: **E, "exact generator posterior" (EGP)**.

## Model E specification
Per machine (type T, signal s), latent and discrete: structure k = (gate lag set G_k, size lag l_k) from a candidate family learned by screening
(all lag sets of size <= 2 from 1..8 incl. "none" x size lag 1..8: 296 candidates, pruned to prior mass >= 1%); size direction d and
win-shift direction e (b_t = 1[payout at t - l_k >= 2]; size state = b_t xor d, win state h_t = b_t xor e; prior P(d, e) learned, so "tied" or
"independent" is decided by data); polarity p; flip time tau (none, or any tick up to the cutoff).
Continuous: mu_on ~ N(a_T + beta_T s, s_on,T), mu_off ~ N(c_T + beta'_T s, s_off,T), integrated on a logit grid.
P(win at t) = sigmoid(mu_side(t) + gamma_T (2 h_t - 1)); side(t) = gate bit xor p xor 1[t >= tau]. Given a win, size ~ V_T[b_t xor d] (two
4-vectors per type). Prediction = exact posterior predictive of the 5-class outcome at the cutoff, summed over all discrete latents.
Hyperparameters: per type (a, c, s_on, s_off, beta, beta', gamma, no-gate mean / sd, V) and global (structure prior, P(d, e)) by maximum
marginal likelihood on flip-free windows (first 45% of long logs, whole observed block of short logs); flip prior (P(no flip) separately for
short and long logs, relative position histogram) by EM on full histories with the rest fixed. No labels, targets or conditions are read.

## Pre-registration (fixed before any result)
Candidates (only these):
- E-ind: EGP, hyperparameters from TRAIN histories only (strict inductive).
- E-trans: identical code, hyperparameters from train + validation + test HISTORIES (permitted: AGENTS rule 2 and DATA.md allow
  histories of other splits; reported separately, compared only with the champion's transductive predictions).
- One blend: 0.5 x EGP + 0.5 x champion, weight FIXED at .5 in advance (no weight search), only if the out-of-sample error
  correlation is below .9.
No hyperparameter grid. Implementation bugs found by the leakage / unit tests may be fixed; modelling changes after seeing validation
numbers are NOT allowed for the promotion decision (any such change is logged as exploratory and not eligible).

Evaluation:
1. Leakage tests first (tests_leakage.py); stop on failure.
2. Train-only episode-grouped 4-fold CV (hyperparameters on 3 folds' histories, outcomes of the held-out fold's queries at cutoffs c, c-1, c-2, c-3
   with history truncated accordingly): Brier / log-loss of the 5-class outcome vs simple baselines. Surrogate only, NOT the official metric.
3. Official Round-1 metric on full validation (inspected before by the team, so not a fresh holdout; EGP fits nothing on it).
   Paired within-condition episode bootstrap vs the champion (same protocol), per condition, per type, top-episode influence, leave-one-out.
4. Runtime: full fit + test prediction under 2 BLAS threads / 1 process, peak RSS, temp disk.

Promotion (all required): leakage tests pass; full-validation gain over the champion in BOTH protocols (inductive vs 58.332, transductive vs 58.355);
paired bootstrap P(gain > 0) >= .95; gain survives removing the top 5 episodes; no condition worse by more than 5%; runtime < 20 min on 2 cores;
reproducible to 1e-8. Otherwise the champion is retained.
