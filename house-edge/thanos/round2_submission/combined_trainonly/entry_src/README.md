# House Edge entry: Thanos E2 (exact generator posterior)

## Entry
- Model: one joint probabilistic model of each machine's payout stream; forecast = exact posterior predictive at the cutoff (structure, size/win-shift direction, polarity, flip time summed exactly; per-machine base win logits integrated on a grid; hyperparameters by maximum marginal likelihood on histories).
- Code version: `thanos/` from the team repository, commit 8e00a50 (egp.py, model.py unmodified), tick set "own".

## Setup
- Python 3.13, `requirements.txt` (numpy, scipy only). No GPU. About 215 MB RAM.
- Released data at `/data` (current round public folder: `train/ validation/ test/`). Main-round data is not used.

## Recreate the forecast
Run from this folder: `python predict.py --data /data --out /output/submission.csv`
- Trains everything from the released histories (train + validation + test histories, observations before each cutoff only; labels, targets and conditions are never read) and predicts the test split; no saved model files, no randomness (deterministic).
- Fits from scratch in several minutes on 2 threads (well under the 30-minute limit); peak memory a few hundred MB. Exact measured figures are in the team's verification log.

## Local check
`python check_submission.py /output/submission.csv --data <public folder>`


## Round 2 variant: correlated on/off rate prior
Thanos (Round 1 code path, adapt=False) + within-machine bivariate-normal prior on the on/off win logits, applied by exact Gaussian conditioning on the other side's own-history posterior (thanos/episode_effect.py: rho_hat, corr_predict). rho per type from TRAIN histories only. Command unchanged.


## + cross-machine recent-win adjustment
After the correlated-prior forecast: P(win) logit += 2.0 * f, f = mean over episode-mates of (win rate over the last 8 pre-cutoff ticks - win rate over all pre-cutoff ticks); size proportions kept (thanos/xm_adjust.py). beta 1.94 = train-only MLE.
