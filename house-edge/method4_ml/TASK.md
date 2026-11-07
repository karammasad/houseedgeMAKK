# TASK: Method 4, gradient boosting with parity features, and blending (owner: Person D)

Read ../AGENTS.md first. Insurance if round 2's mechanism is something Method 1 can't express.
Gradient learners can't discover parities from raw inputs (Shalev-Shwartz et al. 2017), so features carry them.

## What exists
- `features.py`: leakage-safe per-tick features. For each structure (single lags and pairs 1..8): current XOR value,
  full-history agreement rate with the next win, last-40-tick agreement rate; plus lag wins/payouts 1..12, win rates.
  `machine_examples` builds (features, label) for every tick of a history using only earlier ticks.
- `model.py`: scikit-learn HistGradientBoostingClassifier on train-history ticks (subsampled to 150k).
  Score: 48.60 tuning (Method 1: 55.05). Needs `pip install scikit-learn`; cannot ship yet (see AGENTS.md rule 4).

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
