# house-edge

Team repo for House Edge. Read AGENTS.md (rules) and the TASK.md in your method folder.

## Setup (5 minutes)
1. Python 3.11+ (organizers run 3.13). Methods 1-3 need nothing else. Method 4: `pip install scikit-learn`.
2. Unzip the main kit and copy its `public/` folder to `data/main/` (so `data/main/train/queries.csv` exists).
3. Check everything works:
   python -m common.evaluate --method method1_gate --data data/main --audit --note "setup check"
   Expected: tuning 55.054, audit 54.795 (out of 60; the organizers' reference scores 30).

## Current numbers (validation, kit tuning/audit halves)
| method | tuning | audit | note |
| --- | --- | --- | --- |
| method1_gate (default) | 55.05 | 54.79 | the current entry |
| method2_sizes, sizes=dirichlet | 54.88 | 54.56 | size control |
| method3_discovery, max_order=3 | 54.67 | 54.62 | XOR triples hurt short histories |
| method4_ml (default) | 48.60 | not run | baseline gradient boosting |

## Note
`tools/make_entry.py` is written but was not run end to end before handoff. First use: build from method1_gate and confirm the CSV SHA-256 starts with a8d8213e (the uploaded main entry).
