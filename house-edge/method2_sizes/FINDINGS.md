# Method 2 findings: payout sizes given a win

All diagnostics use **train histories only** (`python -m method2_sizes.analysis --data data/main`,
`python3 method2_sizes/feature_search.py`). Scores use `common.evaluate` (kit tuning/audit halves)
with the gate held fixed, so differences come from the size model alone.

## B1: mechanism test (does a win's size depend on earlier payouts?)

**Pooled across machines: no.** A G-test of size × payout at lag k (1..8) and size × previous win's size,
per machine type, against a within-machine shuffle (50 draws) yields 1 of 27 tests at p < 0.05, which is what
chance alone predicts.

**Per machine: yes, weakly.** Summing each machine's own G statistic (size × win/loss at lag k):

| lag | G | shuffled null | p (20 draws) |
| --- | --- | --- | --- |
| 1 | 2156 | 1602 | ≤ 0.05 (above every null draw) |
| 2 | 2069 | 1603 | ≤ 0.05 |
| 3 | 1732 | 1616 | ≤ 0.05 |
| 4 | 1920 | 1557 | ≤ 0.05 |
| 5–8 | ≈ null | | 0.14–0.38 |

So the size mix after a win differs from the mix after a loss, with a **direction that differs by machine**
(the opposite directions cancel in the pooled test). A feature search over single lags and all XOR pairs (lags 1..8)
finds **lag-1 win/loss** to be the best size covariate for 193 of 768 machines. No XOR pair stands out, unlike the
win gate. Part of this signal is probably slow drift in sizes (see below), which also creates a correlation
between sizes and recent wins.

→ Made it a covariate: `LagSizes` (sizes.py).

## Sizes move over time

The z-score of P(1×|win) between first and second halves has var(z) = 1.00 for short history, 1.12 for stable,
1.22 for drift and 1.25 for shared change (1.0 = no change). Sizes drift a little, mostly in drift and
shared-change episodes. → Recency-weighted NPMLE counts (B4).

## B2: episode-wide size shifts — not pursued

Within-episode correlation (ICC) of machines' late-minus-early P(≥2×|win): stable +0.02, drift −0.08,
shared change −0.00. **Machines do not shift sizes together**, so an episode-level size covariate has nothing
to pick up. Not implemented; revisit in round 2 if this ICC changes.

## B3: size change point via the gate's flip posterior — not done

`method1_gate.gate.win_probability` returns only P(win), not the flip posterior. B3 needs Person A to expose it.
Since sizes drift only mildly and B4 already captures most of that, the expected gain is small.

## Results (validation tuning half; baseline NPMLE = 55.054)

| sizes setting | tuning | Δ | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- |
| npmle (current entry) | 55.054 | — | .0738 | .1313 | .0897 | .0646 |
| dirichlet(30) | 54.884 | −0.17 | .0805 | .1363 | .0891 | .0670 |
| B6 minka (per-type fitted α) | 54.990 | −0.06 | .0819 | .1282 | .0894 | .0649 |
| B5 npmle min_wins 20 / 60 | 55.055 / 55.054 | 0 | | | | |
| B4 npmle half_life 64 | 55.131 | +0.08 | .0776 | .1270 | .0878 | .0608 |
| B4 npmle half_life 128 | 55.127 | +0.07 | .0753 | .1288 | .0880 | .0615 |
| B4 npmle half_life 256 | 55.097 | +0.04 | | | | |
| B1 lag [1], strength 8 / 20 / 50 | 54.770 / 55.028 / 55.137 | | | | | |
| B1 lag [1], strength 100 | 55.146 | +0.09 | .0731 | .1282 | .0886 | .0622 |
| B1 lag [1], strength 200 | 55.126 | +0.07 | | | | |
| B1 lags [1,2], strength 20 | 54.662 | −0.39 | | | | |
| **lag [1] s=100 + npmle hl 64** | **55.227** | **+0.173** | .0747 | .1245 | .0874 | .0591 |
| lag [1] s=100 + npmle hl 128 | 55.207 | +0.15 | .0738 | .1261 | .0875 | .0599 |

**Audit half (run once):** candidate 54.851 vs baseline 54.795 (+0.056, same direction) → meets rule 6.

**Full validation, official kit checker** (`kit/check_submission.py --split validation`):
baseline 54.927 → candidate **55.044 (+0.117)**. All four conditions improve
(stable .0744→.0734, short .1317→.1289, drift .0864→.0841, shared .0770→.0738).

## Recommendation

Adopt `{"sizes": "lag", "lag_strength": 100, "npmle_half_life": 64}` as the size model. It's worth about
+0.1–0.17 points, uses only the standard library, is deterministic (byte-identical rerun), and adds
under 1 s of runtime. The remaining size headroom (about +2.5 with true sizes) is mostly hidden state;
nothing else tested here moves it.
For round 2, re-run `analysis.py` and `feature_search.py` first: the lag that drives sizes may change.
