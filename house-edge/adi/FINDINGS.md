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

## C5: short histories (learned cell priors + learned flip prior for short logs)

**Result: tuning 55.676 / audit 55.196** (method 1: 55.054 / 54.795; previous adi: 55.104 / 54.864).
short_history ratio 0.1313 -> 0.0967, and every other condition also improves (stable 0.0738 -> 0.0715,
drift 0.0897 -> 0.0818, shared 0.0646 -> 0.0607). Fit + validation + test predict: 77 s on one CPU.

Settings: `{"max_lag": 12, "and_pairs": true, "prior_flips": true, "prune": 0.01, "floor": 0.05,
"cell_prior": "learn", "cell_group": "type", "p_change_short": "learn"}`

What was done, in order:
1. **Diagnosis** (`python -m adi.c5_decompose`): on short histories, perfect win probabilities would cut the
   ratio to 0.063 and perfect sizes to 0.067. The error is about half gate, half sizes. C5 targets the gate half;
   sizes belong to method 2.
2. **Asymmetric cell prior.** Method 1 gives both gate cells a flat Beta(0.5, 0.5). adi uses an "on" and an
   "off" Beta prior, averaged 50/50 over which cell is on, so polarity flips still just swap cells. The priors
   are fitted on train by maximizing the model's marginal likelihood (coordinate ascent over mean/strength grids).
   - One shared prior: 55.170, and short histories got worse (0.132). The fit sat on the grid edge.
   - Train cell rates differ sharply by machine type (stable train machines, best pair):
     steady on 0.93 / off 0.16, balanced 0.82 / 0.05, jackpot 0.55 / 0.02. Signal shifts them monotonically.
   - Per type (`cell_group: "type"`): **55.428**. Per type x signal: 55.349 (thinner groups, slightly worse).
   - Learned per-type priors: steady on Beta mean 0.95 (k=100) / off 0.15 (k=50); balanced 0.80 (25) / 0.07 (50);
     jackpot 0.60 (25) / 0.02 (100). These match the hand-measured rates, with no condition labels used.
3. **Flip prior for short logs.** TASK suggested p_change 0.7. On tuning, 0.7 is worse (55.33) and 0.3 better
   (55.53). Rather than tune on validation, `p_change_short: "learn"` picks it by marginal likelihood of the
   short train logs: it picks **0.05** -> 55.676. With only the last ~40 ticks observed, a flip is rarely worth
   its prior cost.
4. Not done: inferring missing ticks. Short logs still have ~40 usable windows (the recent consecutive block),
   so imputation would recover only a few more per machine. Low value.
5. `relearn` (re-learning the structure prior under the cell prior): 55.423, no gain. Left off.

## Combined: adi gate (C5) + method 2's GateSizes

`sizes: "gate", gate_cell_prior: "npmle"` uses `method2_sizes.sizes.GateSizes` (Person B's best size model,
imported unchanged). adi's model only changes the win probability and GateSizes only changes the size
split, so the gains mostly add up:

| model | tuning | audit | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- |
| method1_gate (npmle sizes) | 55.054 | 54.795 | .0738 | .1313 | .0897 | .0646 |
| adi C5 gate + npmle sizes | 55.676 | 55.196 | .0715 | .0967 | .0818 | .0607 |
| method 1 gate + GateSizes (method2 best) | 56.038 | 56.377 | .0539 | .1249 | .0655 | .0386 |
| adi C5 gate + GateSizes | 56.703 | 56.808 | .0517 | .0882 | .0577 | .0350 |
| method 1 gate + GateSizes couple+tied (method2 best, fc6b2ab) | 56.999 | 56.974 | | | | |
| **adi C5 gate + GateSizes couple+tied** | **57.685** | **57.427** | .0385 | .0641 | .0368 | .0212 |

- Best: +0.69 tuning / +0.45 audit over method 2's best (couple+tied on method 1's gate), and +2.63 / +2.63
  over method 1. GateSizes' coupling still reads method 1's win-gate posterior internally (it builds its own),
  so the two models never share state.
- Deterministic: two separate runs give byte-identical test CSVs (SHA-256 de89c80d...). Passes
  `kit/check_submission.py`. About 85 s for fit + validation + test.
- Command:
  `python -m common.evaluate --method adi --data data/main --kwargs '{"max_lag": 12, "and_pairs": true, "prior_flips": true, "prune": 0.01, "floor": 0.05, "cell_prior": "learn", "cell_group": "type", "p_change_short": "learn", "sizes": "gate", "gate_cell_prior": "npmle", "gate_couple": true, "gate_couple_iters": 20, "gate_tied": true}' --test-out <csv>`
- Possible next step: feed adi's win-gate posterior (wider family, learned cell priors) into GateSizes'
  coupling instead of method 1's. That needs a hook in method2_sizes, so it's Person B's call.

## Not done yet (TASK.md order)
C4 (L1 logistic fallback gate), C6 (episode-shared lag pairs/polarity), C7 (uniform-vs-learned robustness drill).
Also untested: whether the per-type cell priors transfer to round 2 (they are re-learned from round 2's train
in fit, so nothing is hardcoded).
