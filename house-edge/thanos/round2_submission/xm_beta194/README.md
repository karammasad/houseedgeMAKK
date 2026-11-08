# Round 2 recommended entry: Momin's Thanos + cross-machine win adjustment, beta = 1.94 (train-only)

Identical to `mominsreverseengineering/round2_xm/entry_xm.zip` (Momin; SHA 2ff8b495...) except ONE number in `predict.py`:
beta 2.0 -> 1.94. 1.94 is the maximum-likelihood beta on the 768 TRAIN queries' outcomes (Thanos predictions + the adjustment; no
validation data), so no fitted number comes from validation. Momin's original files are untouched.

| file | sha256 |
| --- | --- |
| `entry.zip` | 245f8a9d5a9f5bfeffd964c40f7e462a2058a49dc96f8e4603ed5eec427e6dac |
| `submission.csv` | pending: produced by a fresh unzip of this exact ZIP with the pinned numpy 2.4.4 / scipy 1.17.1, 2 threads (follow-up commit) |

## Validation (/40; same computation applied to the saved Thanos validation forecasts)
| | plain Thanos | beta 1.94 | beta 2.0 (Momin) |
| --- | --- | --- | --- |
| tuning | 39.061 | 39.149 | 39.149 |
| audit | 39.254 | 39.444 | 39.447 |
| full | 39.183 | 39.317 | 39.318 |

## The three criteria
- Overfitting: LOW (not zero). beta from train outcomes only; the effect is supported by validation-free evidence (Momin's `oos.py`
  history regression: train t 9.3, test t 18.1, placebo ~0; train-only beta 1.94 vs 2.0); audit gained more than tuning; 68/96 episodes.
  Residual: the idea was found after team exploration on validation.
- Leakage: Thanos core byte-identical to the submitted code (`thanos/tests_leakage.py` on round-2 data: ALL 8 PASS); the adjustment uses
  only pre-cutoff ticks of episode-mates (4 automated checks pass: future ticks of every machine ignored, missing != loss, valid, deterministic).
- Runtime: expected ~9 min (Thanos 8.7 min packaged + < 1 s); measured value pending (follow-up commit).
