# adi: Method 3 (structure discovery), Adi's working folder

A drop-in method package for the team evaluator. Run from `house-edge/`:

```sh
python -m common.evaluate --method adi --data data/main --note "..."
python -m common.evaluate --method adi --data data/main --kwargs '{"max_lag": 12, "and_pairs": true, "prior_flips": true, "prune": 0.01}'
```

Standard library only. Settings are described in `model.py`. With default settings it is identical to
`method1_gate` (tuning 55.054).

**Best known (round 4):** method2_sizes `round2.FINAL` with `"size_split": true` added to its adi win_kwargs:
tuning 58.080 / audit 57.832 / full 57.960 on top of the joint-EM FINAL (57.901 / 57.665 / 57.787), keep-check
KEEP. See FINDINGS "Round 4".

**Heads-up: `method2_sizes` (the uploaded main entry, CSV d39f5332...) uses this package as its win model
(`win_model="adi"`).** Any change here must keep existing settings byte-identical. Check with:
`python -c` rebuild of `method2_sizes.round2.FINAL` -> SHA-256 must stay d39f5332... (verified after the round-3
options were added).

| file | what |
| --- | --- |
| `gate.py` | change-point gate generalized to structures = (form, lags): XOR of any order and the AND family (AND/OR/NAND/NOR with input negations); optional flip-aware structure prior |
| `model.py` | `Model` with the wider family, learned prior, prune + floor |
| `c0_analysis.py`, `c0_by_condition.py` | C0 investigation (why triples win discovery) |
| `c5_decompose.py` | C5 diagnostic: how much error is win probability vs payout size |
| `win_diag.py` | where the remaining win-probability error comes from (big misses vs rate calibration) |
| `FINDINGS.md` | results and conclusions |

Current best, adi gate + method 2's GateSizes couple+tied (tuning 57.685 / audit 57.427, about 85 s):
`--kwargs '{"max_lag": 12, "and_pairs": true, "prior_flips": true, "prune": 0.01, "floor": 0.05, "cell_prior": "learn", "cell_group": "type", "p_change_short": "learn", "sizes": "gate", "gate_cell_prior": "npmle", "gate_couple": true, "gate_couple_iters": 20, "gate_tied": true}'`
