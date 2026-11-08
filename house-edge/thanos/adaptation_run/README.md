# Thanos on the adaptation (Round 2) kit

Unmodified Round 1 thanos code, refit from scratch on the adaptation kit's histories (train + validation + test, pre-cutoff only; labels, targets, conditions never read).

    build/venv312/bin/python -m thanos.run --data data/adaptation --splits all --out build/thanos_adapt --also validation

Environment: Python 3.12.13, numpy 2.1.3, scipy 1.18.1, 2 threads. Wall 445.9 s (fit 397.6 s), peak RSS 125 MB (`run.json`).

| file | sha256 |
| --- | --- |
| `submission.csv` (1,536 test queries) | e0228d7b273ba362caac4c4a96eee9947b1346ff462ea0969853685f590412d8 |
| `validation.csv` (768 queries) | 653c06a8672a7d79ab42bab5919483bb8ee2c19f5f42fb26452eae96fd79e350 |

## Validation score (official metric, max 20; `common.evaluate.points`, reference.csv scores exactly 10.00)
| | stable | short_history | drift | shared_change | points |
| --- | --- | --- | --- | --- | --- |
| error / reference | 0.021 | 0.025 | 0.018 | 0.020 | **19.591** |

Kit halves: tuning 19.530, audit 19.627. Mean squared error per query 0.0039 (reference 0.189).
Validation is public and its histories are part of the (transductive) fit, so this is not a guaranteed test score.
