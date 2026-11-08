# How we control and adjust the model

This is the working method, not the model. It's the same discipline Method 2 used (FINDINGS.md, TASK.md status board,
keep checks, reverts kept as options). Anyone on the team can follow it on any method folder.

## The loop

1. **Name one hypothesis.** One sentence: "X depends on Y", with the direction. Add a row to `TASK.md`.
2. **Find the smallest test.** Prefer a diagnostic that reads validation *truth* in an `analysis/`-style script (read only
   for diagnosis) over a full refit. Write what would falsify it before running.
3. **Change one thing.** Keep everything else at the current FINAL. Measure with `run.py score`.
4. **Keep check.** `run.py keep` compares the candidate with FINAL on the same episodes: tuning and audit both up,
   paired bootstrap P(gain) >= 0.95, at least 3 of 4 conditions better, none worse by more than 5%, runtime < 300 s.
   This is the same rule as Method 2's `keepcheck.py`.
5. **Record the verdict in FINDINGS.md** under a new heading: hypothesis, test, numbers, KEEP or REVERT, and why.
   Reverts stay in the file as negative results. They prevent re-running the same idea next week.
6. **If KEEP, promote:** update FINAL in `final.json` (one place), log a status line in TASK.md, rebuild the upload files.

## Rules that never change

- Only edit this folder. Other teams' folders are read, never written to. Copy from them into here if you need code.
- No fit or predict reads validation `targets.csv`, `labels.csv`, or `conditions.csv`. Only `run.py keep` and diagnostics do.
- The audit half is checked once per final candidate, never used to pick between candidates.
- Adoption rule from AGENTS.md rule 6 (+0.15 tuning and an audit move in the same direction) is the team bar. The keep
  check is the stricter, bootstrap-based test we actually use day to day. A change that passes keep but misses +0.15 is
  reported as "below the team bar" with the numbers, not hidden.
- Same input, same CSV: fixed seeds, byte-identical reruns.

## Logs and reporting

- `experiments/ledger.csv` is written by `run.py score`: one row per run, with full-validation points and the four condition
  ratios. This is our log; the team's `experiments/results.csv` is untouched.
- Every reported result goes in the standard table, with a one-row FINAL block on top and history below:

  ```
  | run | tuning | audit | full | stable | short | drift | shared | verdict |
  ```

  `run.py table` prints it from the ledger.

## Round two

Round two uses the same loop. Before the new kit arrives, freeze FINAL and run `run.py score` on main data as a rehearsal.
When the adaptation kit arrives, follow the steps in `README.md`. Changes made for round two get their own FINDINGS entry, with the new-kit numbers.
