# House Edge entry: Thanos E2 (exact generator posterior)

## Entry
- Model: one joint probabilistic model of each machine's payout stream; forecast = exact posterior predictive at the cutoff (structure, size/win-shift direction, polarity, flip time summed exactly; per-machine base win logits integrated on a grid; hyperparameters by maximum marginal likelihood on histories).
- Code version: `thanos/` from the team repository, commit 8e00a50 (egp.py, model.py unmodified), tick set "own".
- Round 2 addition (`xm.py`, applied in `predict.py`): cross-machine win adjustment. For each query, feature f = mean over the OTHER machines in its episode of [their win rate at ticks cutoff-1..cutoff-8, observed ticks only] minus [their overall observed win rate before the cutoff]. Thanos' P(win) is shifted on the logit scale by 2.0 * f; the positive-payout mix is preserved. History-only; beta = 2.0 fixed (chosen on the validation tuning half). Standard library only.

## Setup
- Python 3.13, `requirements.txt` (numpy, scipy only). No GPU. About 215 MB RAM.
- Released data at `/data` (current round public folder: `train/ validation/ test/`). Main-round data is not used.

## Recreate the forecast
Run from this folder: `python predict.py --data /data --out /output/submission.csv`
- Trains everything from the released histories (train + validation + test histories, observations before each cutoff only; labels, targets and conditions are never read) and predicts the test split; no saved model files, no randomness (deterministic).
- Fits from scratch in several minutes on 2 threads (well under the 30-minute limit); peak memory a few hundred MB. Exact measured figures are in the team's verification log.

## Local check
`python check_submission.py /output/submission.csv --data <public folder>`
