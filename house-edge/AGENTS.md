# AGENTS.md: rules for Codex / Claude Code in this repo

House Edge (TAMU Datathon 2026). Predict next-spin payout probabilities [p0, p1, p2, p5, p20] for slot machines.
Score = squared error to the simulator's TRUE probabilities, normalized per condition by a frozen reference.
Full plan: the team's "Slot Machine Odds Challenge: Game Plan" doc.

## Layout
- `common/`            shared loader (`data.py`) and evaluator (`evaluate.py`). Do not change behaviour without telling the team.
- `method1_gate/`      Person A: Bayesian change-point gate model (the current entry).
- `method2_sizes/`     Person B: payout-size models given a win.
- `method3_discovery/` Person C: structure discovery and wider structure families (round 2).
- `method4_ml/`        Person D: gradient boosting on leakage-safe features, blending.
- `tools/make_entry.py` builds a checked, rerun-verified submission from any method folder.
- `kit/`               organizer code (checker, packager, scorer). Never edit.
- `data/main/`, `data/adaptation/`: the kits' `public/` folders. Gitignored. Never commit data.
Each method folder has a TASK.md: read it before working there. Only edit your own folder unless asked.

## Interface every method must keep
`<method>/model.py` defines `class Model`: `Model(**settings).fit(pub)` then `Model.predict(pub, split)`
returns `{query_id: [p0, p1, p2, p5, p20]}` (non-negative, sums to 1). `pub` is a public/ folder path.
`fit` may read `pub/train` fully, and the HISTORIES (not labels/targets) of other splits.

## Commands (run from the repo root)
- Score:  `python -m common.evaluate --method method1_gate --data data/main --kwargs '{"max_lag": 8}' --note "what changed"`
- Audit:  add `--audit` only for a final candidate (once per candidate).
- Entry:  `python tools/make_entry.py --method method1_gate --data data/main --out build/main`
- Check a CSV: `python kit/check_submission.py <csv> --data data/main` (add `--split validation` to score).

## Hard rules
1. No leakage: inputs for a query or a training example use only ticks strictly below its prediction tick,
   for every machine. A label tick is never a feature. Missing ticks (-1 in `seq`) are unknown, not losses.
2. Never fit or tune on the validation audit half; `common.evaluate.folds()` defines tuning/audit halves.
   Never read validation `targets.csv`, `labels.csv` or `conditions.csv` inside `fit` or `predict`.
3. Nothing hardcoded from today's data (lag pairs, priors): learn it in `fit` from `pub/train`. Round 2 changes them.
4. Anything that may ship (methods 1-3) is standard library only. Method 4 may use scikit-learn but cannot ship
   until organizers confirm how extra packages install.
5. Deterministic: fixed seeds; same input -> byte-identical CSV.
6. Adopt a change only if it gains >= 0.15 tuning points and the audit half moves the same way. Log every run
   (the evaluator appends to experiments/results.csv; commit that file).
7. Keep runtime of fit + predict(test) under 5 minutes on one CPU (organizer limit is 30 minutes).
