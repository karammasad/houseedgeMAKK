# TASK: Method 4, gradient boosting with parity features, and blending (owner: Person D)

Read ../AGENTS.md first. Insurance if round 2's mechanism is something Method 1 can't express.
Gradient learners can't discover parities from raw inputs (Shalev-Shwartz et al. 2017), so features carry them.

## What exists
- `features.py`: leakage-safe per-tick features. For each structure (single lags and pairs 1..8): current XOR value,
  full-history agreement rate with the next win, last-40-tick agreement rate; plus lag wins/payouts 1..12, win rates.
  `machine_examples` builds (features, label) for every tick of a history using only earlier ticks.
- `model.py`: scikit-learn HistGradientBoostingClassifier on train-history ticks (subsampled to 150k).
  Score: 48.60 tuning (Method 1: 55.05). Needs `pip install scikit-learn`; cannot ship yet (see AGENTS.md rule 4).

## v2 (current, model.py) - tuning 55.45, audit 55.26 (Method 1: 55.05 / 54.79)
- `model.py`: GBM stacker. P(win) = 0.8 * GBM + 0.2 * Method 1. GBM features (`features2.py`): Method 1 posterior
  summaries at tick t (P(win), P(win|no change), P(change), P(no gate), top-structure mass, change recency),
  the other episode machines' change signals at t, lag wins 1..8, recent win rates. Training ticks: K=24 per
  episode from train + validation/test HISTORIES, plus K_short=24 per long episode with the past masked like
  short_history (40 recent + 24 older ticks). Sizes: `size_hmm.py` (two-state size HMM, EM on train).
- Tuning ratios: stable 0.075, short 0.111, drift 0.087, shared 0.055 (M1: 0.074, 0.131, 0.090, 0.065).
- Runtime: ~150 s fit+test on 8 cores (features in a process pool); about 4x that on 2 cores. Deterministic.
- Old baseline: `--kwargs '{"variant":"v1"}'` (48.60). Sweeps: `python -m method4_ml.sweep --grid '{...}'`.
- Flat (within 0.15, not adopted): GBM grid lr/leaves/min_leaf/l2, K 48, K_short 48, sizes=mix/npmle, size EM on all splits.

## Data findings (method4_ml/analysis/, validation targets read for ANALYSIS ONLY)
- True size distribution given a win takes exactly TWO values per machine type (all 768 validation queries),
  differing mainly in the 1x/2x split; independent of signal, of the win state and of the gate cell.
- That state is only weakly identifiable from history: even with the true distributions and a confident history
  posterior (>0.95), it matches the cutoff state ~77%. Recent wins predict no better than old ones; no lag,
  XOR pair or neighbour explains it. Treat as hidden per-spin noise; EM-learned blurred emissions ~ optimal.
- Error split (Method 1): true P(win) alone -> 57.2, true sizes alone -> 57.6, but the size gain is mostly
  unreachable (see above). Remaining reachable headroom is in the win model, especially short histories.
- True P(win) is bimodal (logit mostly -5..-2 and 1..3), consistent with gated cells plus drift.

## Do, in order
1. D2 Better polarity features: agreement over several windows (20, 40, 80), agreement with the NEGATED XOR,
   time since the agreement last switched sign. Keep if tuning improves by >= 0.15.
2. D3 Residual on Method 1: add Method 1's win probability, size distribution and structure posterior as
   features (import method1_gate; compute for each training tick with ticks < t only).
3. D4 Blend: w * Method 4 + (1 - w) * Method 1, w in {0, 0.1, ..., 0.5}, chosen on the tuning half
   (fit the weight with common.evaluate.points on tuning episodes only).
4. Grid for 1-3: lr {0.03, 0.1}; max_leaf_nodes {15, 31, 63}; min_samples_leaf {50, 200, 1000}; l2 {0, 1, 10};
   average 3 seeds. No class weights or resampling. Train on log-loss (default).
5. Stretch: also train on validation and test HISTORIES (not labels); small sequence model or TabPFN.
