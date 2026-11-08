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

## Round-two night protocol (team decision, from the pasted plan)

1. **Fallback first.** Current pipeline refit on the round-two train data, checked with `kit/check_submission.py`, uploaded.
2. **Retune on the tuning half** only.
3. **Reverse-engineer with train + the tuning half, including that half's `targets.csv`.** Look for general rules about the
   generator (restrictions, menus, tied parameters), never settings for particular episodes. `probes.py` does this and reads
   the tuning half only. A discovery is only a model change if it can be LEARNED inside `fit()` from train/history data.
4. **Audit half once per finished idea.** Keep an idea only if the audit half improves by >= 0.15 points AND no condition gets worse.
5. **Stop new ideas around 6 AM**; submit the best version that passed step 4 (re-run the checker and the fresh-unzip rerun).

Why this does not overfit: every kept change has to move a half it was never tuned on, and the audit is looked at sparingly.
Anything fit on tuning-half targets is allowed to be wrong; the audit gate catches it.

Reading the audit bar honestly: the kit has 12 episodes per condition per half, so a +0.15 audit gain is close to the noise
level of a single half: the same fixed model scores differently on the two halves by 0.02 (v8), 0.09 (v5) and 0.19 (v4) points,
so a +0.15 audit gain can be luck. Treat it as the gate, but not as proof (we have not measured its standard error). The audit gate is the decision rule. As a second,
non-binding view, report `run.py keep` (paired bootstrap on full validation) next to it. If they disagree, default to the
simpler model and say so.

Conflict to be aware of: AGENTS.md rule 6 says "+0.15 on TUNING and the audit moves the same way". The pasted plan says the
bar is +0.15 on the AUDIT half with no condition worse. For this folder we follow the pasted plan (it is the newer decision).
Both must at least move the same way.

## Don't overfit to main

- Anything learned about main stays a hypothesis until the same probe passes on the round-two kit (`probes.py`, `fingerprint.py`).
- Never hardcode values read off main (lags, rates, vectors, sd). Put a value in code only if `fit()` re-derives it.
- A restriction that FAILS on the new kit must degrade gracefully (fall back to the wider model), like the size menu does
  when there are fewer than 50 wins per type.
- See `ROUND2_CLUES.md` for what the organizers have said about round two.

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
