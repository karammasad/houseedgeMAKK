# Thanos on the adaptation (Round 2) kit

Unmodified Round 1 thanos code, refit from scratch on the adaptation kit's histories (train + validation + test, pre-cutoff only; labels, targets, conditions never read).

    build/venv312/bin/python -m thanos.run --data data/adaptation --splits all --out build/thanos_adapt --also validation

Environment: Python 3.12.13, numpy 2.1.3, scipy 1.18.1, 2 threads. Wall 445.9 s (fit 397.6 s), peak RSS 125 MB (`run.json`).

| file | sha256 |
| --- | --- |
| `submission.csv` (1,536 test queries) | e0228d7b273ba362caac4c4a96eee9947b1346ff462ea0969853685f590412d8 |
| `validation.csv` (768 queries) | 653c06a8672a7d79ab42bab5919483bb8ee2c19f5f42fb26452eae96fd79e350 |

## Validation score (official metric, `common.evaluate.points`; reference.csv scores exactly R = 1)
Organizer update: final split is main 60 / adaptation 40 (the kit README's 80/20 is superseded). Points = 40 / (1 + R).
| | stable | short_history | drift | shared_change | points |
| --- | --- | --- | --- | --- | --- |
| error / reference | 0.021 | 0.025 | 0.018 | 0.020 | **39.18 / 40** (R = 0.0209) |

Kit halves: tuning 39.06, audit 39.25 (/40). Mean squared error per query 0.0039 (reference 0.189).
Validation is public and its histories are part of the (transductive) fit, so this is not a guaranteed test score.

## Checks (2026-11-08)
**Overfitting.** Strict inductive fit (hyperparameters from TRAIN histories only; validation never seen): 39.17 / 40 (R = 0.0212), per-condition
ratios 0.021 / 0.025 / 0.019 / 0.020, halves 39.05 / 39.24 -- essentially equal to the transductive 39.18. Nothing was tuned on the adaptation kit
(Round 1 code, unchanged). Test predictions inductive vs transductive: mean max-abs difference 0.005, p99 0.016.

**Leakage.** `python -m thanos.tests_leakage --data data/adaptation`: all 8 tests pass (`leakage_tests.log`): future invariance, cross-machine
independence, fit/predict identical from a copy with labels/targets/conditions/reference deleted, only queries.csv/history.csv opened,
statistics use earlier ticks only, CV folds episode-disjoint, valid 5-vectors. numpy emits matmul RuntimeWarnings (divide/overflow) in every run;
all 2,304 output rows are finite, min prob 5.2e-5, row sums within 4.5e-16 -- not affecting output.

**Runtime / reproduction.** The submitted Round 1 package `thanos/round1_submission/entry.zip`, unzipped into a clean read-only folder, run with an empty
environment (`env -i`) on `data/adaptation`, venv312 (Python 3.12.13, numpy 2.1.3, scipy 1.18.1): 370.2 s wall, 369.1 s CPU, 137 MB peak RSS
(limit 30 min / 16 GiB), output sha256 e0228d7b... -- byte-identical to `submission.csv` here. Not checked: organizer Linux container, kit
`check_submission.py` (blocked locally).
