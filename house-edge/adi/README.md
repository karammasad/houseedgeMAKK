# adi: Method 3 (structure discovery), Adi's working folder

A drop-in method package for the team evaluator. Run from `house-edge/`:

```sh
python -m common.evaluate --method adi --data data/main --note "..."
python -m common.evaluate --method adi --data data/main --kwargs '{"max_lag": 12, "and_pairs": true, "prior_flips": true, "prune": 0.01}'
```

Standard library only. Settings are described in `model.py`. With default settings it is identical to
`method1_gate` (tuning 55.054).

| file | what |
| --- | --- |
| `gate.py` | change-point gate generalized to structures = (form, lags): XOR of any order and the AND family (AND/OR/NAND/NOR with input negations); optional flip-aware structure prior |
| `model.py` | `Model` with the wider family, learned prior, prune + floor |
| `c0_analysis.py`, `c0_by_condition.py` | C0 investigation (why triples win discovery) |
| `FINDINGS.md` | results and conclusions |
