# Findings

Ledger of hypotheses, tests and verdicts. Newest at the top. Each entry: hypothesis, test, numbers, verdict, why.
Reverts stay here as negative results. See CONTROL.md for the keep rule.

## Entry 0: baseline (v8 as copied)

- Hypothesis: none. This is the reference every later keep check compares against.
- Test: `python3 run.py score --name "v8 baseline"` on `../data/main`.
- Expected (from Kareem's log, e46c44d): tuning 58.309, audit 58.287, full 58.297. Verdict: reference.
