# Method 5: simulation-based / amortized predictive inference (independent research branch)

Status: COMPLETE. Result: stop development (best model 36.0/60; no complementary signal). See `RESULTS.md` and `FEASIBILITY.md`. Nothing here imports or modifies Methods 1-4.
Rules: stdlib at inference; torch/numpy only offline for training with exported weights; deterministic; no validation
or test labels in `fit`; episode-grouped CV; the audit half is final confirmation only.

| file | what |
| --- | --- |
| `FEASIBILITY.md` | no organizer simulator (repo 404); reconstruction fidelity; why synthetic supervision cannot add information; plan |
| `data_checks.py` | read-only checks: split disjointness, strict cutoff, short-log layout, example counts, noise floors |
| `RESULTS.md` | final tables: competition scores, CV, ablations, complementarity, deployment checks, verdict |
| `data.py`, `audit.py` | causal window/static builder; leakage audit (0/300 future-perturbation leaks per split) |
| `cv.py`, `cv_ablate.py`, `cv_seed.py`, `report.py` | episode-grouped CV of B0/B1/B2 and the GRU variants S1/S2/S3/S3b, seed controls, report |
| `train_final.py`, `model.py`, `bench.py`, `weights/` | offline training, standard-library inference, one-CPU benchmark and parity, exported weights |
| `frozen_eval.py`, `complement.py`, `audit_eval.py` | competition-metric scoring, complementarity vs the team models, the single audit run |
| `test_method5.py` | unit tests (feature parity, no future dependence, missing ticks flagged, valid deterministic outputs) |
