# adi / Method 3 findings

## C0: why XOR triples win discovery but don't help the model

**Answer: most winning triples are flip-invariant aliases of the machine's true XOR pair.**
They beat a fixed-polarity pair only because they cancel polarity flips. Method 1 already models flips with
its change point, so in the model the triples add noise and nothing else.

Reproduce: `python -m adi.c0_analysis --data data/main` and `python -m adi.c0_by_condition --data data/main`
(train split; structure fitted on the first 70% of each long history, scored on the last 30%).

- 576 long-history train machines: the best structure is a pair for 480, a triple for 69 and a single lag for 27.
- Of the 69 triple winners, 68 share a lag with the best pair, and 40 are the exact alias pattern:
  pair (i, j) -> triple (j, i+i', i+j') where (i', j') = (i, j). Example: (4,7) -> (7, 8, 11).
- Mechanism: if w_t = XOR(w_{t-4}, w_{t-7}) XOR f(t), where f is the polarity, then
  w_{t-4} = XOR(w_{t-8}, w_{t-11}) XOR f(t-4). Substituting gives
  w_t = XOR(w_{t-7}, w_{t-8}, w_{t-11}) XOR f(t) XOR f(t-4). The polarity cancels except for 4 ticks
  around a flip, so the triple works across a flip, at the cost of compounding two noisy gates.
- Held-out per-spin log-likelihood:

| condition | best pair (fixed polarity) | best triple | best pair refit on held-out window |
| --- | --- | --- | --- |
| stable | **-0.295** | -0.440 | -0.270 |
| drift | -1.502 | -0.537 | **-0.374** |
| shared_change | -1.532 | -0.544 | **-0.342** |

  On stable machines the pair beats the triple. On flipping machines the fixed pair collapses and the triple
  survives, but a pair with the correct polarity beats both, which is what Method 1's change point provides.
- So selection bias isn't the main cause, and neither is dependence on another machine. Discovery without a
  flip model is biased toward triples on drift/shared machines. Any structure search has to model flips
  (adi's `prior_flips=True` does this when learning the structure prior).

## C2/C3: wider structure family with a learned, pruned prior

Logged in `experiments/results.csv` (notes starting with "adi"). Validation, kit tuning/audit halves, out of 60:

| settings (`--method adi --kwargs ...`) | tuning | audit | short_history ratio |
| --- | --- | --- | --- |
| `{}` (= method1_gate) | 55.054 | 54.795 | 0.1313 |
| `{"prior_flips": true}` | 55.088 | | 0.1287 |
| `{"max_lag": 12}` | 54.811 | | 0.1503 |
| `{"and_pairs": true}` | 55.073 | | 0.1299 |
| `{"max_lag": 12, "and_pairs": true, "prior_flips": true}` (342 structures, no prune) | 54.462 | | 0.1788 |
| **same + `"prune": 0.01, "floor": 0.05`** | **55.104** | **54.864** | **0.1274** |

The stable, drift and shared_change ratios are identical in every row. Only short histories react to the family size.

- The learned prior is very sparse. Over 342 candidates (XOR singles/pairs at lags 1..12, plus 264 AND-family
  pairs), three XOR pairs carry about 85% of the weight: (1,3) 0.30, (2,5) 0.28, (4,7) 0.27. No AND/OR form gets
  above 0.005, and only 3 structures exceed 0.01. Main-round machines use XOR gates on these three pairs only.
- A wide family without pruning hurts short histories: the uniform floor spreads mass over hundreds of
  structures that aren't real. Pruning at 0.005–0.02 with any floor (0–0.1) recovers it completely. The results
  are flat (all 55.104), so the setting isn't fragile.
- Gain over method 1 is +0.05 tuning / +0.07 audit. That's below the 0.15 adoption bar, so **don't swap the main
  entry for it**. Its value is round 2: the wide, pruned family finds whatever gates train uses, with nothing
  hardcoded, and costs nothing on main data. If round 2 uses AND/OR gates or lags 9–12, method 1's fixed family
  (XOR, lags 1..8) can't represent them, and this one can.
- Round-2 command (refit only):
  `python -m common.evaluate --method adi --data data/adaptation --kwargs '{"max_lag": 12, "and_pairs": true, "prior_flips": true, "prune": 0.01, "floor": 0.05}' --test-out <csv>`
  Runtime: about 31 s for fit + validation predict on one CPU.

## Not done yet (TASK.md order)
C4 (L1 logistic fallback gate), C5 (short-history priors / p_change for 64-tick logs), C6 (episode-shared lag
pairs/polarity), C7 (uniform-vs-learned robustness drill).
