# TASK: Method 4, gradient boosting with parity features, and blending (owner: Person D)

Read ../AGENTS.md first. Insurance if round 2's mechanism is something Method 1 can't express.
Gradient learners can't discover parities from raw inputs (Shalev-Shwartz et al. 2017), so features carry them.

## What exists
- `features.py`: leakage-safe per-tick features. For each structure (single lags and pairs 1..8): current XOR value,
  full-history agreement rate with the next win, last-40-tick agreement rate; plus lag wins/payouts 1..12, win rates.
  `machine_examples` builds (features, label) for every tick of a history using only earlier ticks.
- `model.py`: scikit-learn HistGradientBoostingClassifier on train-history ticks (subsampled to 150k).
  Score: 48.60 tuning (Method 1: 55.05). Needs `pip install scikit-learn`; cannot ship yet (see AGENTS.md rule 4).

## v7 (current, model.py): tuning 58.260, audit 58.242, FULL 58.249
v7 = v6 (bag of 5 tick samples) + a 50/50 blend with a structured offset-logistic correction of P(win) (`logit_stack.py`: size-state x P,
size-state x (1-P), P(change) x (2P-1), per machine type; gate logit fixed as the offset; l2=1). Keep-check vs v6: KEEP (P .986, 4/4; keepcheck.py criterion). Vs team FINAL (58.156 / 57.986 / 58.074): +0.104 tuning / +0.256 audit /
+0.175 full: still below the +0.15 TUNING bar of AGENTS rule 6 (audit moves the same way), so a team decision is needed to adopt it.
See RESEARCH_LOG.md (experiments 18-24) for what else was tested; main-round error is near the ceiling.

## v5 (previous, model.py): tuning 58.236, audit 58.143, FULL VALIDATION 58.190 (kit checker 58.189711 / 60)
Team best before this (adi size_split + Momin FINAL sizes): 58.080 / 57.832 / 57.960. v5 gain: +0.156 tuning / +0.311 audit
against the OLDER team best (58.080 / 57.832) it was +0.156 tuning; against the CURRENT team FINAL (58.156 / 57.986 / 58.074) it is
+0.080 tuning / +0.157 audit / +0.116 full, which does NOT meet AGENTS rule 6 (+0.15 tuning AND audit same direction). Do not describe it as
passing that rule without a team decision to change the rule. v4 was 57.956 / 57.768 / 57.865.
Pieces:
1. Win gate: adi's model with `size_split` (round 4: win rate depends on the hidden size state), imported read-only.
2. NEW, signal-shifted cell priors (`signal_shift.py`): data finding (analysis/onrate_structure.py) - inside (type, size state) the
   true on-cell win rate shifts with `signal` (jackpot ~+0.09 per step) and its spread drops to sd .08-.11 (vs .164 per type).
   adi's per-type prior is shifted by (beta_on, beta_off) x signal on the log-odds scale; betas learned per type by marginal
   likelihood on TRAIN histories (identical choice when learned on all histories): balanced (.3, .2), jackpot (.3, .2),
   steady (.4, .4); evidence gain 39-94 nats. Gate alone: 58.080 -> 58.156.
3. Sizes: Momin's FINAL GateSizes (learned tie weight + joint EM), `SIZES` in model.py.
4. NEW, stacker sees the hidden size state: features3 adds the size model's P(1x|win) at tick t (and its type-centred value),
   so the GBM learns how the size state moves the win rate. 58.175 -> 58.234.
5. Stacker: HGB lr .03, 15 leaves, min_samples_leaf 1000, 3 seeds, blend .4; K=6 + K_short=6 ticks per episode (same accuracy
   as K=24, 4x cheaper features).
KEEP-CHECK vs team FINAL (method2_sizes round2.FINAL incl. adi size_split + cell_signal), `python -m method2_sizes.keepcheck`:
  A FINAL: 58.156 / 57.986 / full 58.074; B v5: 58.236 / 58.143 / full 58.190; bootstrap P(B better) .987, 4/4 conditions,
  none worse, 241 s -> keepcheck.py verdict KEEP (that tool's own bootstrap criterion, NOT the +0.15 tuning rule of AGENTS rule 6,
  which v5 misses: +0.080 tuning vs FINAL). (adi independently found the same signal slope, +0.30, in round 5.)
Runtime: fresh fit + test with 2 workers 310 s (organizer limit 30 min); deterministic (cached == fresh, b23618aa). Needs scikit-learn (rule 4). Package with `python -m method4_ml.make_entry`.
Tried this round, not adopted: size stacker for P(1x|win) (hurts: 58.236 -> 58.196 at weight .6), HGB grid of 72 configs (flat,
58.17-58.24), K 6/12/24 (flat), pooled histories for sizes (flat), ramp/gradual drift (true rates are discrete even in drift:
<=4% mid-range), removed the dead per-structure cell-stat features.

## v4 (superseded): tuning 57.956, audit 57.768, FULL VALIDATION 57.865 (kit checker: 57.865046 / 60)
Base = adi's win gate + Momin's FINAL size model (method2_sizes.GateSizes: couple + tied with LEARNED tie weight + joint EM,
settings in model.SIZES), both imported read-only. Alone this is Momin's 57.901 tuning (reproduced exactly with blend=0).
On top: HistGradientBoosting stacker on gate-posterior features (features3.py), lr .03, 15 leaves, min_samples_leaf 1000, 3 seeds,
P(win) = 0.4 * GBM + 0.6 * gate; trained on K=24 ticks/episode from train + val/test HISTORIES plus K_short=24 masked
short-history copies. Stacker gain over Momin's base: +0.055 tuning (below the 0.15 rule: the base does nearly all the work).
Ratios (full validation): stable .030, short .063, drift .029, shared .027. Runtime ~160 s fit + test on 8 cores, deterministic.
Needs scikit-learn (cannot ship until organizers confirm, rule 4). Round 2: tie weight is now LEARNED (no hand-set .95 / .7 risk).
PACKAGING: tools/make_entry.py omits adi/ and requirements.txt. Use
`python -m method4_ml.make_entry --data data/main --out build/main_m4` (packages adi/, pins the scikit-learn stack, cache off).
Tried and flat (<= +0.04, not adopted): pooling train+val+test histories for the size model (no change) or gate priors (+0.035),
mixture-of-Betas cell priors (57.69), signal-aware cell means, per-structure cell stats, offset boosting, averaging gate variants,
all gate settings (within +-0.03), GateSizes couple_mix / couple_alpha / strength / joint_iters (within +-0.01 on the new base),
GBM grid (lr .01-.1, leaves 7-31, min_leaf 100-3000, l2 1-10).
Data checks (analysis/): no structure sharing inside an episode (33.1% vs 33.5% chance), neighbour machines add nothing to the win
gate, win gate is XOR of win bits only, episode rate ICC .07-.09, splits consistent (no leakage/duplicates/shifts; ~1% constant or
empty histories). Remaining error: per-machine rate noise (jackpot "on" cells) and short-history size level.
With the true P(win) the score would be ~59.4; with true sizes ~58.3 (both measured on the older base).

## v2 (superseded; 55.45 / 55.26): GBM stacker on Method 1 + two-state size HMM
## Data findings (method4_ml/analysis/, validation targets read for ANALYSIS ONLY)
- True size distribution given a win takes exactly TWO values per machine type (all 768 validation queries),
  differing mainly in the 1x/2x split; independent of signal, of the win state and of the gate cell.
- CORRECTION (v3): my first pass called this state unpredictable. Wrong. Method 2 (method2_sizes/FINDINGS.md) showed it
  follows the payout CATEGORY (loss / 1x / big) at one earlier lag per machine; my tests only used win/loss lags and XOR
  pairs, so they missed it. method2_sizes.GateSizes (couple + tied) captures it; v3 uses it. My two-state HMM
  (size_hmm.py, history-only, no lag) is kept only as an option (sizes="hmm"/"mix"); it is ~2 points worse.
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
