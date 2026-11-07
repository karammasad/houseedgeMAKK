# TASK: Method 3, structure discovery for round 2 (owner: Person C)

Read ../AGENTS.md first. Round 2 (9 PM) changes "which temporal relationships recur". Your job: make round 2 a refit.

## What exists
- `discover.py`: per machine, best structure among single lags / XOR pairs / XOR triples up to --max-lag, with a
  complexity --penalty; prints how often each wins.
  `python -m method3_discovery.discover --data data/main --split train --max-lag 12 --max-order 3`
- `model.py`: Method 1 with `max_lag`, `max_order`, `prune` (drop structures with learned prior weight < prune)
  and `floor` (uniform mixing) settings.
- Results: default (lags 1-8, pairs) 55.05. With XOR triples (max_order=3) 54.67 tuning / 54.62 audit: short
  histories get worse (0.131 -> 0.161), long ones unchanged. Uniform prior over lags 1-12 earlier: 51.1 audit.
- OPEN QUESTION: discover.py picks XOR triples as best for about half the machines even at penalty 10
  (e.g. (4,5,7), (3,4,10), (7,8,11)), yet triples don't help the model. Find out why first (selection bias over
  220 candidates? a triple equal to a pair on most ticks? dependence on another machine?). Write FINDINGS.md.

## Do, in order
1. C0 Resolve the open question above (held-out ticks: fit structure on the first 70% of a history, score on the rest).
2. C1 Make sure every prior is learned in fit() from pub/train only (true now; keep it true).
3. C2 Extend discovery: gate forms single, XOR-2, XOR-3, AND, OR; lags 1..12; report per split; < 2 minutes.
4. C3 Sparse prior over the wide family: prune {0.005,0.01,0.02} x floor {0.05,0.1}. Keep if within 0.1 of default.
5. C4 Fallback gate for unknown forms: per-machine L1 logistic regression on lag wins 1..12 and pairwise XORs,
   recent ticks weighted (half-life {64,128}); stdlib implementation (simple coordinate descent).
6. C5 Short histories: stronger learned priors; p_change {0.5,0.7} for 64-tick logs.
7. C6 Do machines in one episode share lag pairs or polarity more than chance? If yes, add an episode-level prior.
8. C7 Robustness drill: prior learned from train vs uniform; within 0.6 points means round 2 is safe.
9. Tonight 7:45 PM: dress rehearsal of the round-2 runbook on main data from a clean clone.
