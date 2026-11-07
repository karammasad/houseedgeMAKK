# Method 5 results: learned sequence representations vs hand-designed structure

Verdict: **stop development. Not a candidate for the main entry, and no complementary signal.** Method 5's best
model scores 36.0/60 on validation (organizer baseline 30; Method 2 58.074; team best ~58.2).

## Competition score (exact five-category formula; frozen models trained on TRAIN only)

| model | tuning | audit | full validation | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| organizer baseline | n/a | n/a | 30.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| B2 boosting + last 8 outcomes | 33.700 | not run | not run | .7229 | .8986 | .7512 | .7490 |
| S1 GRU dense (ablation reference) | 33.518 | not run | not run | .7143 | .9271 | .7667 | .7522 |
| **S2 GRU observed-only (selected)** | **35.931** | **36.103** | **36.009** | .5917 | .7827 | .6798 | .6107 |
| Method 2 final (verified) | 58.156 | 57.986 | 58.074 | .0211 | .0617 | .0257 | .0242 |
| team best available (Kareem's v5 via Adi) | 58.244 | 58.142 | 58.192 | .0217 | .0542 | .0272 | .0175 |

S2 was selected on tuning + CV evidence (CV tie with B2; tuning +2.2). The audit was run once, on S2 only
(`audit_eval.py`); it agrees with tuning (35.93 -> 36.10), so the number is stable, not a selection artefact.
Condition ratios are error relative to the organizers' frozen reference (lower is better).

## Episode-grouped 5-fold CV on train (held-out labelled cutoff queries, Brier; n = 768, 96 episodes)

| model | pooled Brier | vs B0 prior [95% CI] | vs B2 [95% CI] |
| --- | --- | --- | --- |
| B0 type x signal prior | .5803 | n/a | +.047 [+.034, +.059] |
| B1 boosting, window frequencies (no order) | .5547 | -.026 [-.035, -.017] | +.022 [+.015, +.029] |
| **B2 boosting + last 8 outcomes** | **.5330** | -.047 [-.060, -.036] | n/a |
| S1 GRU dense, seed 0 / seed 1 | .5379 / .5391 | -.042 / -.041 | +.005 / +.006 (CIs include 0) |
| S2 GRU observed-only, seed 0 / seed 1 | .5284 / .5378 | -.052 / -.043 | -.005 / +.005 (CIs include 0) |
| S3 GRU columns shuffled | .5630 | -.017 | +.030 [+.019, +.042] |
| S3b GRU categories shuffled, mask fixed | .5639 | -.016 | +.031 [+.020, +.042] |

## What was learned
1. **Order of outcomes matters, independent of missingness.** S1 - S3b = -.026 [-.037, -.015]; S3b keeps the
   observation mask and gap pattern fixed and removes only the order, and equals S3 (which also moves the gaps).
2. **The sequence encoder adds nothing over a simple feature model.** All four GRU runs (.528-.539, mean .5331) equal
   B2 (.5330); on the 59,760 late-tick examples B2 beats the second-seed GRUs by .008 [.006, .010].
3. **S2's apparent CV edge was seed luck:** the same architecture gives .5284 (seed 0) and .5378 (seed 1).
   Gap information does not help (dense S1 is no better than observed-only S2).
4. **Far from the structural models.** Excess error versus the TRUE probabilities (per episode): Method 5 .13-.16,
   team v5 .006 (~25x). A learner given one-hot outcomes cannot infer each machine's hidden gate from ~200 ticks.
5. **No complementary signal.** Against v5 and Method 2: Method 5 beats the team in 0/48 tuning episodes; its error
   vectors correlate +.28-.33 with the team's; the optimal blend weight beta is NEGATIVE (-.018 to -.027, CI excludes
   zero on the wrong side) and every fixed blend (w = .05 ... .50) lowers the score (-.17 ... -9.4). Its disagreements
   with the team point away from the truth because it shares the team's bias toward type averages without having the
   structure.
6. **Synthetic supervision was set aside** (see FEASIBILITY.md): no organizer simulator (repository returns 404), and a
   reconstruction's posterior predictive is what the existing Bayesian models already compute.

## Deployment checks (S2, pure standard library, one CPU thread)
- fit (load 154 KB weights) 2 ms + predict 9.4 s for 1,536 test queries (budget 300 s); 768 validation queries 4.6 s.
- Reruns byte-identical (sha256 4997e019... test, 11da6124... validation); 0 invalid rows; peak memory 22 MiB.
- Parity with the torch model on 40 validation queries: max |difference| 1.6e-07. (S1: 10.2 s, parity 1.2e-07.)
- Imports in `model.py`: json, math, os, operator, `common.data`. Nothing from Methods 1-4. Training (offline, torch, fixed
  seeds) takes ~2 minutes on one CPU: `python -m method5_simulation.train_final --mode nogap --out method5_simulation/weights/s2.json`.
- B2 (scikit-learn) was not exported to standard-library trees: it is below the bar and not the selected model.

## Reproduce
`python -m method5_simulation.audit` (leakage audit) | `cv --fold k` x5, `cv_ablate`, `cv_seed`, `report` (CV) |
`train_final`, `bench --tag s1|s2`, `frozen_eval`, `complement <cache.json>`, `audit_eval` | `python -m unittest method5_simulation.test_method5`.
