# Round 2: Thanos + cross-machine win adjustment (challenger, verification in progress)

## Change
Post-processing of unchanged Thanos E2 predictions (`xm.py`, called from `entry_src/predict.py`; Thanos `egp.py` / `model.py` byte-identical to `deploy_thanos/entry_src`).
For each query: f = mean over the OTHER machines in its episode of
[their win rate at ticks cutoff-1 .. cutoff-8, observed ticks only] - [their overall observed win rate before the cutoff].
Thanos' P(win) is shifted on the logit scale by beta * f with beta = 2.0 fixed; the 1x/2x/5x/20x mix is preserved.
History-only (queries.csv + history.csv). Standard library only. Adds < 1 s to runtime.

## Validation (Round 2 kit, official checker, Thanos validation predictions = Adi's `thanos/adaptation_run/validation.csv`)
| episodes | Thanos | + adjustment | diff (20-pt scale) |
| --- | --- | --- | --- |
| all 96 | 19.5913 | 19.6592 | +0.068 |
| tuning 48 | 19.5304 | 19.5744 | +0.044 |
| audit 48 | 19.6271 | 19.7237 | +0.097 |

Condition loss-ratio change (all / tuning / audit): stable -12/-7/-17%, short_history -11/-12/-10%, drift -39/-34/-42%, shared_change -9/+3.1/-31%.
68 of 96 episodes improve; dropping the 5 most helpful episodes leaves about half the gain.

## Selection history (read this before trusting the number)
Explored on validation: P(win) temperature (7 values, all worse), own/common blend (2 weights, ~0), within-episode residual correlation,
recent-rate features (null), cross-machine lag correlations (printed for tuning AND audit), then two pre-specified forms
(A: lags 1-8, B: lag 3) x 6 betas on tuning. Chose A, beta = 2 (B was slightly better on tuning but its lag was picked after seeing audit).
So audit is NOT an untouched holdout. Independent label-free evidence (`oos.py`, histories only, never used for selection):
peer recent wins predict a machine's next win after controlling for its own base and recent rates:
train +0.18 (t=9.3), test +0.24 (t=18.1); placebo with another episode's peers ~0.

## Files
- `entry_src/`, `entry_xm.zip`: the challenger package. `validation/xmA_validation.csv`: the scored validation CSV.
- `xm.py` (feature + adjustment), `xm_write.py` (apply to a CSV), `cmp2.py` (official metric, tuning/audit, episodes), `oos.py`, `calib.py`, `idea.py`, `run_r2.py`.
- Test CSV and fresh-unzip reproduction: pending, added in a follow-up commit.

Edge case: a peer with zero observed ticks before the cutoff would divide by zero; impossible on this kit (min 64 observed ticks).
