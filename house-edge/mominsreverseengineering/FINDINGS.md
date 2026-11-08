# Findings

Ledger of hypotheses, tests and verdicts. Newest at the top. Each entry: hypothesis, test, numbers, verdict, why.
Reverts stay here as negative results. See CONTROL.md for the keep rule.

## Entry 17: feasibility check of the shared-base + cook pool (overfitting, leakage, 2-core runtime)

**Leakage: PASS for cook.** `experiments/leak_cook.py`: Adi's current `adi.cook` fitted and predicted on a copy of the kit with EVERY answer file deleted (only queries.csv and history.csv of train / validation / test kept) with a file-open audit hook:
the run completed, only those six files were opened, and validation and test predictions equal the full-kit run exactly (max difference 0.0, which also shows a fresh fit reproduces the earlier predictions). Static scan: the only label read on cook's import path is the neighbour-adjust option (`adi/model.py:309`), never enabled by its configuration. Our shared-base model: PASS (Entry 12).
**Overfitting: inconclusive; the pool's benefit is thin and concentrated.** Weight fixed at .5 in advance (no tuning). Pool vs plain v8 +0.0708 (96 episodes, 95% CI [+0.013, +0.136]) but the top 5 episodes carry 66%, the top 10 98%, and without the top 10 the gain is +0.001; 62 / 96 episodes better.
Pool vs OUR shared-base model +0.0142: top 5 carry 192%, without the top 10 -0.032, 51 / 96 better, tuning +0.024 / audit +0.004. Adi's own v8 + cook pool shows the same shape (+0.030; top 5 carry 116%, without the top 10 -0.023, 53 / 96).
Cook alone is no better than v8 (tuning +0.0075, audit -0.0094). The pooled gain is jackpot-driven (the 48 most jackpot-heavy episodes +0.073, the 48 least -0.003). Dozens of variants have been judged on this one 96-episode validation set across the team, so each +0.02 to +0.05 step probably overstates the true gain.
**Runtime (30 minutes, 2 cores): feasible ONLY as a saved-checkpoint entry.** Measured, `experiments/time_cook_2core.py` (saved cook model, 2 workers / 2 threads, hard kill at 29 min): load 0.1 s + predict(test) 196.8 s = 197 s wall, 192 CPU-seconds, peak memory 44 MB, output identical to the earlier test CSV (0.0).
Ours with a checkpoint: packaged predict 31 s (Entry 14). A pooled checkpoint rerun is therefore about 3.8 minutes (31 s + 197 s, sequential). Fit side: cook fit + predict 1069-1102 s on 8 cores (Adi: 527 s on his laptop); ours about 17 min fresh on 8 cores; so roughly 20-35 min of fitting inside the Round 2 window, once.
WITHOUT a checkpoint (refit in the organizers' rerun) it is infeasible: ours alone was 27.2 min on 2 cores and cook's fit is of the same order again.
**Not tested:** a combined package in ONE process. Static check: cook patches `_split_mixture`, `machine_tables`, `win_probability_grid` and GateSizes `_win_posterior` / `fit` / `predict`; it saves the original `_split_mixture` at import, so in a combined package it falls back to our rewritten version (same signature, equal to the original within 3e-14); our xsg path uses a separate function.
`adi/gate.py`, `adi/model.py`, `method2_sizes/sizes.py` and `common/` are unchanged on main since our snapshot, so the combination is a merge of cook's newer files with our edits. An identity check (pooled predictions equal the separately computed members) is still needed; estimated 1-2 h plus one cook fit.
**Verdict.** Technically feasible with checkpointing (about 4 min rerun), leak-free, but the evidence that it improves on our shared-base model alone is weak. Cheapest way to benefit: offer our shared-base model to Adi as a pool member (his selector, checkpoint builder and exact-pins environment already exist) instead of building a second pipeline here.

## Entry 16: pooling our shared-base model with Adi's cook (his round-14 recipe): best validation score so far, but the increment over ours is small and not distinguishable

**Source.** Adi's round 14 (origin/main `ed85b6b`): a fixed 50/50 linear pool of v8 and his standard-library `adi.cook` (r2 inner, adi size coupling, size menu) gives 58.337 / 58.321 / 58.327 (tuning / audit / full), saved as a checkpoint entry that reruns in 132 s. He did not pool our shared-base model. Run here from a scratch copy of `main` (his files untouched): `experiments/run_cook_main.py`, 1069 s for fit + validation + test.
**Sanity (reproduces his numbers):** cook alone tuning 58.318, v8 + cook pool tuning 58.338, full 58.328 (his 58.327). Weight fixed at 0.5 in advance (not tuned; tuning-half weights .25 / .5 / .75 give 58.348 / 58.361 / 58.358, flat).

| model | tuning | audit (consumed, info only) | full (kit scorer) | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v8 baseline | 58.311 | 58.289 | 58.2981 | .01704 | .05671 | .02242 | .02060 |
| v8 + shared-base (ours) | 58.338 | 58.378 | 58.3547 | .01548 | .05485 | .02290 | .01955 |
| adi.cook | 58.318 | 58.279 | 58.2973 | .01861 | .05507 | .02275 | .02041 |
| v8 + cook pool (Adi's recipe) | 58.338 | 58.322 | 58.3281 | .01754 | .05501 | .02212 | .01998 |
| **shared-base + cook pool (new)** | **58.361** | 58.382 | **58.3689** | .01662 | .05387 | .02223 | .01906 |
| new pool vs ours | +0.024 | +0.004 | +0.0142 | +7.4% | -1.8% | -2.9% | -2.5% |

Kit scorer on the pool: 58.368919 / 60. Paired episode bootstrap (96 validation episodes): pool vs ours +0.014, 95% CI [-0.033, +0.061], P(better) .72, better in 51 of 96 episodes (tuning half alone: +0.024, CI [-0.034, +0.082], P .80);
pool vs plain v8 +0.071, 95% CI [+0.013, +0.136], P .993. Pooling adds about +0.03 on top of v8 but only about +0.014 on top of ours: the two gains overlap (about 80% additive: 0.057 + 0.030 predicted, 0.071 observed).
Stable gets worse (cook is weaker there: .0186 vs .0155); short, drift and shared improve. A condition-gated pool was already falsified by Kareem (#42, #43), so the weight stays fixed.
**Verdict.** The pool is the best validation score so far (58.3689) and is credibly better than plain v8, but the increment over our shared-base model alone is inconclusive and below the +0.05 tuning bar set in advance, so it is NOT adopted on its own evidence; it is consistent with Adi's independent +0.03.
**Not done / risks.** No pooled checkpoint entry exists for this combination: cook patches `adi.gate` / `GateSizes` in-process and our folder's gate carries the xsg and optional state-shift edits (`_split_mixture` is touched by both), so combining them in one process needs an isolation and identity check (about 1-2 h); cook alone took 1069 s here (Adi: 527 s).

## Entry 15: win-informed size direction (the one approved experiment): STOP by the pre-stated rule; effect real in direction, too small to adopt

**Idea.** Win outcomes carry information about the size direction d (win rate differs by size state, shift gamma = .46 / .53 / .30), but the size model updates d from sizes only; win history reaches it only through the lag prior P(k).
**Derivation (no double counting).** For a size lag k: P(d | w, y, k) is proportional to P(d) * [sum over structures and flip times of P(s)P(tau) L_win(w | s, tau, d)] * L_size(y | k, d). The size model already uses win data once, through P(k); the hook only replaces its constant direction prior
(for the {big} partition, only where an xsg structure with that size lag exists) by sum_w w * sigmoid(logit(dl) + lam * LLR), with weights normalised WITHIN the structures sharing lag k, so k is never re-weighted and the prior is added once. Rates, on/off roles and polarity are integrated out as in the gate.
Each tick's win/loss enters once (L_win) and each win's size once (L_size); past payouts are conditioning only. Code: `adi/gate._shared_dir_ll`, `method4_ml/shared_base.win_direction_evidence`, optional hook in `method2_sizes/sizes.py` (`win_dir_lam`, off by default).
**Verification** (`experiments/verify_wd.py`): lam = 0 vs the ORIGINAL committed size model max |diff| 0.0 over 2,304 queries; per-direction win likelihood vs brute force 1.4e-14; product form vs the exact joint posterior over d 7.0e-14 over 540 cases (including direction priors .3 / .7). Hook is live when on (lam .5 moves predictions by up to .10).
**Training-only held-out diagnostic** (`wd_diag.py`; 576 long train machines, 72 episodes, 25,392 pseudo-query rows, targets at 45-58% of the log, prefixes n40 / real short layout / full, lambda in {-.5, 0, .25, .5, 1} chosen on one episode half and scored on the other; 330 s on 8 cores).
Rule fixed in advance: proceed only if on n40 and short64 the cross-fitted size log-loss improves with an episode-clustered 95% CI below 0, the shuffled-history control shows no gain, and the five-category squared error does not worsen.

| info | size log-loss change (cand - base) | 5-class sq err change x1e-4 | SHUFFLED control: size log-loss | control: 5-class x1e-4 |
| --- | --- | --- | --- | --- |
| n40 | -0.00182 [-0.00419, +0.00029] | -4.12 [-9.19, +0.39] | +0.00083 [+0.00015, +0.00160] | +2.56 [+0.98, +4.44] |
| short layout | -0.00114 [-0.00346, +0.00104] | -2.76 [-7.64, +1.83] | +0.00057 [-0.00017, +0.00135] | +2.15 [+0.58, +3.93] |
| full | -0.00212 [-0.00380, -0.00044] | -3.50 [-7.07, -0.17] | +0.00040 [-0.00017, +0.00099] | +1.02 [-0.19, +2.35] |

Learned lambda +1.0 in both halves (the Round 1 sign link holds; negative lambda lost). By type at the short layout all intervals include 0 (balanced -.0038, jackpot +.0006, steady +.0002).
**Verdict: STOP (rule not met).** The direction is right and consistent and the shuffled control gets WORSE (a wrong machine's win history hurts), so the evidence is machine-specific and not an artifact; but the gain is about 0.1-0.2% of size log-loss and the interval on both short settings includes 0.
Final five-category error does not worsen. No official tuning run was made; no audit was used. The hook stays in the folder as an off-by-default option (lam = 0 reproduces the shipped model exactly). Not worth adopting on this evidence; a larger sample (the Round 2 kit) is the only way to tell +0.1% from 0.
Runtime of this experiment: verification < 1 min, diagnostic 330 s (the first run also took 306 s before a control-baseline bug of mine, fixed and rerun with the same protocol). Fit-time cost of the hook in the full pipeline was NOT measured because no full run was made.

## Entry 14: deployment fix (checkpoint packaging): rerun 27 min -> about 30 s, predictions unchanged

Problems found in `method4_ml/make_entry.py` (our copy): (1) the packaged entry refits from scratch in the organizers' rerun (27.2 min on 2 cores, Entry 13); (2) packaging fits TWICE (CSV, then the fresh-unzip check);
(3) it refused to build unless installed packages equalled a stale pin file (scikit-learn 1.9.1 vs 1.8.0 here) and `--allow-mismatch` would ship pins that did not produce the CSV;
(4) it looked for `kit/` under this folder, which does not exist (crash after the long fit); (5) `workers` came from `os.cpu_count()`, which a 2-core container may report as the host's.
Fix (no change to model maths or predictions): fit once or reuse a model from `adapt.py` (`--model-pkl`), ship `model.pkl` in the ZIP, `predict.py` loads it with 2 workers / 2 threads and falls back to a full refit if the checkpoint is missing or unreadable;
requirements are written from the installed packages that produced the model (warning, not failure, versus the team pin file); kit path resolved; adapt.py saves `model.pkl` per candidate.
**Verified on main with the candidate** (`--use-cache` for the fit only): packaged predict.py 31 s, fresh-unzip rerun 30 s, byte-identical (max difference 0.0), identical to the in-memory fitted model (0.0) and to the earlier test CSV (sha256 f907a388...);
official checker VALID (1536 queries); ZIP 3.1 MB (model.pkl 8.1 MB); the whole build took 1 min 56 s with the cached fit (a fresh fit adds one 17-27 min fit, once, instead of twice).
Fallback tested: with a corrupted model.pkl predict.py prints a warning and starts refitting. NOT tested: the pinned package versions (the ZIP pins the installed 1.8.0 / 2.4.4 / 1.17.1, which must be installable by the organizers), and a fresh no-cache end-to-end build.
Also fixed: the deployment script printed megabytes as gigabytes (peak memory is about 240 MB main process, 95 MB largest worker).
Round 2 flow: `adapt.py --stage full-select` (fits each candidate once, saves model.pkl) -> `python3 -m method4_ml.make_entry --data <kit> --out <dir> --model-pkl <chosen>/model.pkl` (no refit).

## Entry 13: deployment feasibility (official 2-core / 30-minute rerun): MARGINAL as packaged; a saved checkpoint is the fix

**Official limits** (`kit/README.md`, `RULES.md`, `SUBMISSION_README.md`): 30 minutes, two CPU cores, 16 GiB RAM, no network, `/tmp` 128 MiB, `/output` 32 MiB, output within 1e-8 of the submitted CSV.
They apply ONLY to the organizers' offline rerun of the packaged `reproduce.json` command, which checks prize contenders after the cutoff. The score comes from the uploaded CSV. No runner code ships in the kit.
**Existing log:** Kareem measured plain v8 cold on 2 workers / 2 BLAS threads at 1104 s (his machine); his widened-gate v8 took 1516 s and his 58.314 pool about 38 min, both rejected on runtime.
**One test** (`experiments/deploy_test.py`; fresh fit, cache off, test predictions, 2 workers, 2 BLAS/OpenMP threads, hard kill at 29 min; macOS has no CPU affinity, so cores are limited by thread/worker counts):
v8 + shared-base finished in **1632 s (27.2 min)**, margin 2.8 min (9%). Stages: feature building 1098 s (67%), size-menu fit 260 s, gate fit 222 s, test predict 32 s, size model 21 s, shared-base structure fit 16 s.
CPU-seconds 2980 (so no 2-core machine can beat about 1490 s). Peak memory about 0.24 GB main process, 0.10 GB largest worker (the printout mislabelled MB as GB). Output identical to the 8-thread run (max difference 0.0).
**Not measured:** plain v8 on THIS machine at 2 cores (so the shared-base overhead cannot be separated from machine speed: Kareem's 1104 s on his machine versus 1632 s here); the pinned dependency versions (local scikit-learn 1.8.0 / numpy 2.4.4 / scipy 1.17.1 vs pins 1.9.1 / 2.1.3 / 1.18.1).
**Verdict:** under the limit here, but with a margin too thin to rely on the organizers' hardware being as fast. The packaged entry also sets `workers` from the detected core count (capped at 8), which oversubscribes a 2-core container that reports host cores.
**Minimal fixes that do not change predictions (not implemented):** (1) force `workers=2` and 2 BLAS/OpenMP threads in `predict.py`; (2) ship a saved fitted model and load it in `predict.py` (the guide endorses "load a saved checkpoint"): rerun time falls to about a minute, predictions byte-identical, requirements must pin the exact versions used to save it.

## Entry 12: trust audit of v8 + shared-base (leakage, overfitting, reproduction): leakage PASS; improvement INCONCLUSIVE, leaning mildly positive

**Audit status: the audit half is CONSUMED.** Candidate frozen first (predictions saved 00:53, committed 00:55 as 178bba6, ledger row had audit/full blank), then audit/full were computed once on request. Do not tune, route or blend using audit results.
**Official scorer, full validation (/60):** v8 baseline 58.298069, v8 + shared-base 58.354723 (+0.056654); same 768 ids, same order, five categories p_0,p_1,p_2,p_5,p_20, vectors sum to 1.
Tuning 58.3107 -> 58.3376 (+0.027), audit 58.2885 -> 58.3783 (+0.090).
**Leakage: PASS.** Static: the only label read on the model path is the neighbour-adjust option (`adi/model.py:309`), never enabled by v8; `common/evaluate.py` (the only reader of targets / conditions / reference)
is not imported by the model; the stacker's labels are history ticks before each cutoff (`method4_ml/model.py` `_features`: `range(t_min, cutoff)`); features use `seq[:t]` (`features3.py`); no hardcoded ids or answers.
Runtime: a Python audit hook over a full no-cache fit + both predicts (`experiments/audit_io.py`) shows ONLY `queries.csv` and `history.csv` of train / validation / test were opened; no labels, targets, conditions or reference.
Caveats: (1) the stacker, menu and logistic correction train on validation and test HISTORIES, including the same machines later predicted: allowed (AGENTS rule 2: histories only) and uses only earlier ticks, but it is not out-of-fold at the episode level;
(2) the runtime trace ran on the synthetic kit (same code path); (3) the idea lineage saw validation truth: Adi's size-state finding used both halves, my probes used the tuning half, and the rejected state-shift run showed audit numbers (before this candidate existed).
**Overfitting assessment (96 episodes, paired): INCONCLUSIVE.** Gain +0.0566, 95% CI [-0.0054, +0.1202], 90% CI [+0.0044, +0.1102], P(gain > 0) = .963. Only 53 of 96 episodes improved (sign test p = .18).
The gain is concentrated: the 5 best episodes carry 79%, and without the 10 best the gain is -0.018; the 48 most jackpot-heavy episodes contribute +0.072, the 48 least -0.016 (consistent with the mechanism, but test-set jackpot share matters).
Per condition (ratio change, 95% CI, points): stable -9.1% [-20, +8] +0.022; short -3.3% [-7.5, +1.3] +0.026; shared -5.1% [-13, +2] +0.015; drift +2.1% [-6, +10] -0.007 (immaterial; improved on the audit half, worsened on tuning). Every condition interval includes 0.
**Reproducibility:** rerun (cached features) gives identical validation predictions (max abs diff 0.0), tuning 58.3376, and a test CSV that passes the official checker (1536 queries, sha256 f907a388...). A fresh no-cache rerun of the candidate on main was NOT done (about 17 min); the fresh runs on the synthetic kits and the original fresh run agree with the cached path.
**Verdict:** credible mechanism, no leakage found, small and concentrated gain: INCONCLUSIVE as a Round 1 upgrade. Keep plain v8 as primary; `shared_base=True` is an optional configuration.
**Round 2 technical readiness (not proven performance):** derives lag pairs from the fitted prior (`derive_pairs`), refits every population parameter and the structure weights from the new train data, off by default, fallback = plain v8, `adapt.py --stage full-select` selects on the official tuning metric.
Remaining risks: the size state is assumed to be big-category vs the other two (direction averaged); the population fit assumes polarity is constant in the first 45% of a long log; at most 6 pair families, gates of two lags; fresh fit about 17 min on 8 cores and NOT measured on the organizers' 2 cores / 30 min limit (the packaged entry retrains; a saved checkpoint may be needed); needs scikit-learn/numpy/scipy pins (local scikit-learn 1.8.0 vs pinned 1.9.1).

## Entry 11: the Round 2 selection rule was scoring the wrong objective; fixed (full-pipeline selection)

Main-kit rehearsal of `adapt.py --stage select --no-guard` (audit untouched): 3 min 49 s; gate fit 205 s, structure fit 17 s; it CHOSE the new structure on its
gate-level win-side metric (mean ratio -9.3%: stable -23%, short -4%, drift -1%, shared -15%). The full pipeline on the same kit gave only +0.027 tuning points
with drift worse (Entry 10): v8's stacker and logistic correction already use the size state, so a gate-only comparison overstates the gain.
Same on the synthetic changed kit: win-side -26% at the gate, but only +0.086 official tuning points (/40) for the full pipeline.
Fix: `--stage full-select` runs plain v8 and v8 + structure as two full pipelines in parallel (4 workers each), chooses on the official tuning metric (margin
0.05 points on a 60-point scale, scaled by the kit's max_points; no condition worse than 10%), one audit look after the choice; plain v8 is the fallback and is
always produced. On main this rule keeps plain v8 (+0.027 < 0.05).
Also fixed: a kit-checker failure crashed `final` and threw away a finished fit (found when the synthetic kit's metadata failed the checker); it is now reported
in final.json (`checker_ok`) and the run continues. The synthetic kits now carry checker-valid metadata (round adaptation / 40, `synthetic: true`).

## Entry 10: integrated prototype (`xsg` structure in the v8 gate) through a real tuning evaluation: +0.027, NOT distinguishable from noise

**Build.** `adi/gate.py` (our copy): new structure form `xsg` = the six cells of `xs` with ONE base rate per gate side (independent logit-normal priors,
on/off roles averaged) and a +-gamma size-state shift (direction averaged); vectorised over flip times; flip times only remap cell counts upstream, so
change-point inference is untouched. Checked against an independent brute-force implementation (max difference 3e-14) and falls back exactly to `xs`
when its parameters are absent. `method4_ml/shared_base.py`: derives candidate pairs and size lags from the gate's FITTED structure prior, fits the
population per type by marginal likelihood (two starts, L-BFGS-B restarts), records convergence and bound hits, adds one `xsg` per pair and relearns the
structure prior with flips marginalized. Off by default (`shared_base=False`).
**Disabled == v8:** gate P(win) vs the original gate code on 2,304 validation+test queries: max difference 7e-15; full pipeline tuning 58.3107 (= baseline).
**Fit on main train (18 s):** pairs derived (1,3)/1, (2,5)/2, (4,7)/4; all three types converged; gamma .46 / .53 / .30, slopes .32-.40, on-side sigma .26-.34;
bound hits: jackpot off-side sigma at its .2 floor, and nuisance terms (structure-weight logits, the 'no gate' component). New structure takes 42% of the prior.
**Flips-enabled screening** (`screen_shared.py`, in-sample priors for both): V1 - V0 = -5.5 to -6.7 x1e-4 pre-flip, -7.0 to -9.0 post-flip (CIs exclude 0);
stable, drift and shared all negative; jackpot -14 to -19, balanced -4 to -12, steady about 0 (+1.8 to +2.5 post-flip).

**Frozen tuning evaluation** (official metric, tuning half only; the audit half was never computed):

| model | tuning /60 | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- |
| v8 (current code, structure off) | 58.3107 | .02035 | .05276 | .02723 | .01555 |
| v8 + xsg | 58.3376 | .01854 | .05198 | .02827 | .01519 |
| change | +0.027 | -8.9% | -1.5% | +3.8% (worse) | -2.3% |

Paired episode bootstrap on the tuning half (48 episodes): +0.027 points, 95% CI [-0.066, +0.117], P(better) = .735; episodes better: stable 7/12, short 8/12,
drift 6/12, shared 7/12. Runtime: fresh fit + validation 1012 s (v8 fresh runs took 966-1197 s; the new step itself is 18 s).
**Verdict: NO-GO for adoption as the Round 1 model.** The gain is well below the noise between halves and drift gets worse, so it does not earn the single
audit look. Kept as an OPTIONAL, data-selected candidate for Round 2 (adapt.py adopts it only when the new kit's tuning half shows a clear gain).

**Round 2 adaptation procedure (`adapt.py`), tested:**
- synthetic kit with CHANGED lag pairs (2,4)/(3,6)/(1,5) and gamma .5: derived the new pairs, recovered gamma .54 / .50 / .47 and slopes, chose the new
  structure (win-side mean ratio -26%), passed the audit guard; 159 s.
- synthetic kit with the size shift REMOVED: learned gamma .16 / .02 / .00, chose plain v8 (one condition +11% worse); 203 s.
These test the mechanics only; the synthetic generator is mine, not the organizers'.

## Entry 9: size-shift ablation (`prefixdiag_ablation.py`): the size-state shift is ESSENTIAL; pooling alone is worse than flip-fair v8

Same folds, pseudo-queries, targets and flip-fair v8 as Entry 8 (HD - V0ff reproduced exactly). HDg0 = HD with gamma fixed to 0 (verified 0 in every fit), all else identical.
Paired difference (x1e-4, negative = first model better, 95% CI clustered by episode, 72 episodes):

| information | HD - V0ff | HDg0 - V0ff | HDg0 - HD |
| --- | --- | --- | --- |
| 10 ticks | -8.07 [-13.89, -2.09] | -5.98 [-11.50, +0.17] | +2.09 [-0.80, +5.18] |
| 20 ticks | -4.44 [-10.68, +1.45] | -0.02 [-3.32, +3.30] | +4.42 [-0.67, +10.16] |
| 40 ticks | -6.96 [-12.77, -1.31] | +3.34 [-0.52, +7.34] | +10.30 [+3.90, +17.22] |
| real short layout | -8.03 [-13.73, -2.43] | +3.78 [-0.06, +7.81] | +11.81 [+5.15, +18.69] |
| 80 ticks | -8.26 [-13.78, -3.07] | +7.11 [+3.04, +11.75] | +15.37 [+8.42, +22.94] |
| full history | -8.62 [-13.84, -3.37] | +8.86 [+4.87, +13.04] | +17.48 [+10.01, +24.71] |

Jackpot machines, HDg0 - HD: n20 +16.5 [+0.1, +33.7], n40 +30.6 [+10.0, +52.2], short layout +33.2 [+12.0, +55.2], full +40.1 [+16.2, +64.4]; jackpot HDg0 - V0ff is POSITIVE
(worse than flip-fair v8): n40 +9.7 [+1.1, +19.7], full +19.7 [+9.5, +31.0]. Balanced: the shift matters from 40 ticks up (HDg0 - HD +6.1 at n40, +7.4 short layout, +14.7 full).
Steady: nothing at 20+ ticks (CIs include 0), but at 10 ticks the steady gain (-21.0) is kept almost entirely without the shift (-19.4): a pure prior effect.
Optimizer status (L-BFGS-B, 120-iteration cap): HD converged in folds 0, 1, 3 and NOT in fold 2 or the full-data fit; HDg0 converged in folds 0, 2, 3 and the full fit and NOT in fold 1.
Fixing gamma gave fewer iterations (47-80 vs 72-120) but one non-converged fit in each model family's folds (HD 1 of 4 folds, HDg0 1 of 4): no reliable convergence benefit.
In-sample (full data): gamma adds +159.9 log-likelihood for 3 parameters (the gate-level state-shift evidence earlier summed to about +156).
Interpretation by the pre-stated rules: the gain DISAPPEARS without the shift from 40 ticks upward, so the size-state effect is essential there; at 10 ticks about three quarters of the
gain survives without it (prior effect, CI includes 0). The ablation never outperforms HD. Pooling the six sub-cells under one shared base rate is NOT sufficient: with gamma = 0 it is worse
than flip-fair v8, whose mixture of plain and split structures captures part of the state effect. The two production-relevant facts together: (a) the earlier production state_shift run
(independent sub-cells, loose strengths, shift added) LOWERED the score; (b) shared base + shift + tight priors WINS in the flip-free window. Untested: whether pooling or tightness matters
once the shift is present (would need independent sub-cells with the same shift and strengths).

## Entry 8: flip-fair control (`prefixdiag_flipfair.py`): a real but smaller gain SURVIVES; conditional GO to ONE ablation, not to a tuning evaluation

Same pseudo-queries, prefixes, targets, folds and scoring code as Entry 7 (reproduced exactly: V0 Brier and HD - V0 identical in all six settings).
V0ff = v8 with its flip prior set to 1e-9 on every path. Verification: V0ff vs a computation that passes no flip times at all: max difference 9.1e-9 over
4,480 sampled rows (original v8 differs from no-flip by 0.0037 on average). Flip prior used by original v8: .05 for n10 / n20 / n40 (100% classed short),
.5 for n80 and full (0% short), and a mix for the real short layout (17% short): this confirms the Entry 7 confound.
Paired difference (x1e-4, negative = first model better; 95% CI clustered by episode; 72 long-log episodes, 8,464 pseudo-queries per setting):

| information | V0ff - V0 (flip handicap) | HD - V0 (Entry 7) | HD - V0ff (FAIR) | HA - V0ff (FAIR) | gain that disappears |
| --- | --- | --- | --- | --- | --- |
| 10 ticks | -0.28 | -8.35 | -8.07 [-13.89, -2.09] | -7.38 [-12.82, -1.49] | 3% |
| 20 ticks | -1.36 | -5.80 | -4.44 [-10.68, +1.45] | -3.81 [-9.91, +1.68] | 23% |
| 40 ticks | -1.54 | -8.51 | -6.96 [-12.77, -1.31] | -5.83 [-12.03, -0.10] | 18% |
| 80 ticks | -14.71 | -22.98 | -8.26 [-13.78, -3.07] | -7.08 [-12.62, -1.81] | 64% |
| real short layout | -14.55 | -22.58 | -8.03 [-13.73, -2.43] | -6.86 [-12.46, -1.37] | 64% |
| full history | -14.16 | -22.78 | -8.62 [-13.84, -3.37] | -8.76 [-14.00, -3.36] | 62% |

By type, HD - V0ff: jackpot carries it (-21 to -24 at 40 ticks, real short layout and full, CIs exclude 0); steady is about 0 (+0.3 to +1.3, CIs include 0) except
n10 (-21); balanced is negative but its CIs include 0 (n10 -8.1, n40 -4.4, short layout -5.1, full -7.3).
Numerical validity: 3 of 8 cross-fits did not converge (HA folds 0 and 3, HD fold 2, all at the 120-iteration cap); HD steady rho sat at its bound in the full fit.
The gain does NOT depend on the unverified correlation: HA (no rho) gives -6.9 to -8.8, within about 1e-4 of HD.
Reading: about 62-64% of the Entry 7 gain at n80 / short layout / full was the flip handicap; the remainder (about -8e-4 on P(win)) is real in this window, significant
in 5 of 6 settings (not at 20 ticks), and largest for jackpot. Still NOT deployable: no change-point model, pairs hardcoded, no deployable flip-aware form yet; it can
only help flip-free machines (stable and short-history), not drift or shared change. Caveat: the gain at FULL history is large for a pure prior effect, which points to
structure (pooling the six sub-cells under one shared base rate per side), not tighter strengths.
Next single ablation (NOT run): HD with the size shift forced to zero (the fitting code already supports `gam_fixed=0`). If the gain stays, the source is shared base
rates / tighter priors and the failed `state_shift` run was the wrong half of the idea; if it vanishes, the shift is the source. It does not separate pooling from tightness
or from the structure restriction; those would be later, separate ablations.

## Entry 7: hierarchy fits and the prefix diagnostic (train only, cross-fitted): NOT PROVEN, comparison is biased toward the alternative

**Seven population fits** (`hierarchy.py`, 768 machines, rates integrated out on a logit grid; log-likelihoods A -19269.7, B -19268.4,
C -19244.1, D -19243.0, E -19241.4, F -19258.2, G -19261.8). Numerical validity: A and B hit the iteration limit; C ended BELOW nested D and E
(impossible at a true optimum, so C is stuck); boundary solutions in the full fits: HA jackpot-off strength 400 (my bound), HC steady-on sigma and
jackpot-off sigma at the .15 lower bound, correlation rho at its bound (tanh .995) for balanced and steady, HD steady rho at its bound. Convergence
flags were NOT recorded for the 12 cross-fitted fits. So the seven models are NOT ranked; only coarse conclusions are drawn: on-cell Beta
strengths 62-182 (v8's grid-capped 25-100), spread and signal slopes not rejected as shared across types, size shift per type, rho unverified.

**Prefix diagnostic** (`prefixdiag.py`: 576 long machines, 8,464 pseudo-queries, targets = realized wins at ticks 45-58% of the log, inputs only before
the target, priors cross-fitted over 4 episode-grouped folds; V0 = v8 production gate; H* = grid predictors with shared base rate + size shift):
paired Brier difference vs V0 (x1e-4, negative = better than V0; 95% CI clustered by episode)

| information | V0 Brier | HA | HC | HD |
| --- | --- | --- | --- | --- |
| n10 | 1478.6 | -7.7 [-13.1, -1.8] | -8.1 [-13.8, -2.2] | -8.4 [-14.2, -2.4] |
| n20 | 936.6 | -5.2 [-11.4, +0.3] | -5.5 [-11.9, +0.4] | -5.8 [-12.0, +0.1] |
| n40 | 880.4 | -7.4 [-13.4, -1.6] | -7.4 [-13.6, -1.1] | -8.5 [-14.5, -2.7] |
| n80 | 884.1 | -21.8 [-31.1, -13.0] | -21.5 [-30.9, -12.7] | -23.0 [-32.4, -14.0] |
| short64 | 891.3 | -21.4 [-29.6, -13.9] | -21.5 [-29.7, -13.6] | -22.6 [-30.6, -15.0] |
| full | 878.8 | -22.9 [-31.1, -14.7] | -21.5 [-29.5, -13.3] | -22.8 [-30.8, -14.6] |

short64 by type (HA / HC / HD): balanced -17.6 / -18.5 / -19.4; jackpot -37.8 / -38.4 / -39.5; steady -12.5 / -11.4 / -12.6.
ORACLE columns (structure fixed to the full-history MAP pair; NOT deployable): V0o short64 -10.0 [-16.9, -3.5], HCo -27.3. The V0o numbers at n10 / n20
(-523 / -61) are implausibly large and were not investigated; do not use them.
Decomposition of V0's short64 Brier (V0 vs itself, so unaffected by the confound below): data scarcity +12.5, structure ambiguity +10.0 (x1e-4).

**Why this is NOT a result.**
1. The comparison favors H by construction. H has no flip model; my window is flip-free by construction (flips sit near 68% of the cutoff), but V0
   carries its flip prior: .5 when it classifies the log as long, .05 when short. By adi's rule ("short" = more than half the ticks missing) n10 / n20 / n40
   are short (flip prior .05) while n80, full and about half of short64 (t below ~128 with 64 observed) are LONG (flip prior .5). The pattern fits: the
   long-classified rows show -21 to -23, the short-classified ones -5 to -8. This is an inference from the logic; the rows were not saved, so I did NOT verify it.
2. Plausibility: a -22.9e-4 gain at FULL history is about half the entire five-class squared error of a stable machine on validation (about .0035-.0041),
   and it appears at full information where priors should matter least. That is not credible as a real prior effect.
3. H is not deployable as built: no change-point model, hardcoded pairs (1,3), (2,5), (4,7), size lag = smaller lag, 45% window, 2-state direction mixture.
4. The diagnostic cannot separate tighter priors from the size shift from the shared base rate from rho: every H variant includes all of them.
**Verdict: NO-GO for a tuning evaluation as it stands.** The cleanest signals are the short-classified rows (n10, n40: about -7 to -8 with CIs excluding 0),
which are the least affected by the flip handicap, but they are not yet shown to survive a fair V0.
Single cheap fix (not run): rerun V0 only with its flip prior set to ~0 in this window (about 3 ms x 50k predictions), then compare again.

## Entry 6: state_shift (Entry 4's extension) built and scored in full: REJECTED

Gate-only fit chose gamma = 0.4 / 0.6 / 0.4 (balanced / jackpot / steady) with train log-evidence gains +47.5 / +90.3 / +18.5, peaked
and unimodal. The full v8 score then went DOWN (the run had already finished when the pivot said not to launch it; recorded as is):

| run | tuning | audit | full | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v8 baseline | 58.3107 | 58.2885 | 58.2981 | .01704 | .05671 | .02242 | .02060 |
| B: state_shift on | 58.2594 | 58.2667 | 58.2609 | .01568 | .05735 | .02414 | .02223 |

Change: tuning -0.051, audit -0.022, full -0.037. Stable improved (-8%); short, drift and shared got worse. Both halves fell, so the decision
does not rest on the audit number. Lesson: a clean train-evidence gain (up to +90 nats) did not transfer to squared error, and the model
(shift on both sides, big-vs-rest grouping, structure prior NOT relearned under the shift) is mis-specified somewhere. Not adopted.
Code stays in the copy as an off-by-default option (`state_shift=False`); nesting at gamma = 0 verified to 3e-14.

## Entry 5: sign-tie experiment (Entry 4) RUN: INCONCLUSIVE, underpowered (neither confirmed nor falsified)

`python3 signtie.py --data ../data/main` (train histories only, 192 stable machines, no targets, no audit). Two runs, both reported:

| run | machines identified from first half | main on-cell mean D (95% CI) | N1 decoy | N2 other machine | N3 time-shift | N4 permuted |
| --- | --- | --- | --- | --- | --- | --- |
| pre-registered design (size lag free, threshold 5 nats) | 14 (13 evaluable) | +0.36 [-0.35, +1.22] | -0.07 | +0.23 | +0.04 | +0.35 |
| exploratory follow-up (`--tie-k`: size lag = smaller gate lag) | 11 (10 evaluable) | +0.61 [-0.10, +1.65] | +0.11 | +0.13 | +0.33 [+0.12, +0.56] | -0.02 |

Deviations, both made before any D was seen: the size-gain threshold went from 8 to 5 nats (the first draft identified 2 of 192);
the `--tie-k` run is a second, EXPLORATORY look and is not part of the pre-registered result.
**Verdict under the pre-registered rule: NOT PASSED** (N2 +0.23 is above the +-0.15 band and the CI contains 0). Read honestly:
- The direction and size match the truth-based finding (+0.61 vs the +0.7 to +0.9 logit gap seen in the tuning-half truth), but with 10 machines
  the CI is [-0.10, +1.65]: it cannot separate +0.6 from 0.
- The controls are not clean nulls at this n: N4 (an estimator null) gave +0.35 in one run and -0.48 / +0.44 in balanced / steady in the other,
  and the time-shift control is +0.33 with a CI above 0 (a persistent state or lag echo is possible). Per-machine D is too noisy here.
- Side result that DID hold: the size lag equals the smaller gate lag for 11 of 11 identified machines in the tied run, and the pairs
  recovered are again exactly (1,3), (2,5), (4,7).
- **The bottleneck is identifying the size state, not the win-rate shift.** Only 6-7% of machines get a decisive state from about 45 wins;
  short logs have 15-20. That argues the payoff of the proposal is smaller than the +0.05 to +0.1 guess in Entry 4: the shift only helps a
  machine whose state direction is known, and a hard threshold understates what a soft posterior extracts (Momin: short-log state is wrong 26%).
Next step, if pursued (not run): pool all 576 long machines with SOFT size-state weights from the existing size model, instead of a hard
threshold on 192; that is a model-based test, so it is no longer training-only-and-model-free.

## Entry 3: Candidate A (gate capped at lag 8): NO-OP (recorded, not adopted)

| run | tuning | audit | full | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v8 baseline | 58.3107 | 58.2885 | 58.2981 | .01704 | .05671 | .02242 | .02060 |
| A: `gate.max_lag = 8` | 58.3107 | 58.2885 | 58.2981 | .01704 | .05671 | .02242 | .02060 |

Identical in every column. Cause (read from the cached gate fits of both runs): after the learned-prior pruning (`prune=0.01`) the
gate keeps exactly **7 structures**, with 0.0000 prior mass on any lag above 7, for `max_lag` 12 and for 8 alike:
`xs(1,3,1) .183, xs(2,5,2) .169, xs(4,7,4) .158, xor(1,3) .156, xor(4,7) .147, xor(2,5) .140, none .048`.
So the lag-pair support (three pairs, each plain and size-split) is already INFERRED from train, and the official "lags up to eight"
statement adds nothing. Verdict: no change; no further lag-range sweeps.

## Entry 4: map of v8's generative assumptions, and the one missing mechanism (analysis only, nothing implemented)

**Fixed in code** (changeable only by editing config or code): win bit = payout > 0 (1x counts as a win); a gate is the XOR (or one of
4 AND-masks) of at most two win bits; candidate pool = all pairs to lag 12; ONE optional polarity flip per machine that swaps the two
cells and nothing else; flip time uniform on a grid (step 2, from tick 40 to cutoff-3), flip probability .5 on long logs and a learned
.05 on short logs; "on"/"off" assignment averaged 50/50; each cell rate constant in time with an INDEPENDENT Beta prior per cell;
a "short" log = more than half the ticks missing; a tick whose gate inputs are missing is dropped from cell counts, and the cutoff cell
is mixed by observed share when its inputs are missing; size split uses the payout category (loss / 1x / big) at the SMALLER lag only.
**Inferred from data in `fit()`**: which structures exist (330 candidates pruned to 7), their prior (flip-aware EM), Beta cell priors
per type with a learned log-odds slope in `signal`, short-log flip prior, size-state levels, win-pair-to-size-lag coupling and tie
weight, the exact 2-vector size menu per type, the tree stacker and logistic correction.
**Changeable without new algorithm code**: `GATE` and `SIZES` dicts, `size_split` mode, `cell_group`, blend weights, prune/floor.
**Needs new code**: any dependence between sub-cell rates, a joint win x size likelihood, more than one flip.

**The missing mechanism: one hidden binary state drives BOTH the size vector and the win logit, with a type-independent sign.**
Tuning-half truth (`probes.py --probe state`, audit half not read): in every one of the six type x gate-mode cells the low-1x (bigger
payout) size state has the HIGHER win rate: balanced off .081 vs .035, on .890 vs .792; jackpot off .021 vs .009, on .700 vs .501;
steady off .209 vs .117, on .962 vs .941. In logit units the gap is about 0.7-0.9 in five of the six cells (steady on 0.46). The state
explains 18-40% of the within-(type, mode) variance of true P(win) (37% pooled; 68% on short-history episodes, n too small to trust).
The model uses `xs`/`xst` cells, whose six (or four) sub-cell rates have EXCHANGEABLE Beta priors: same mean, no sign, no link to the
size evidence. A short log leaves roughly 5 observations per sub-cell, so the model cannot tell which sub-cell rate applies at the cutoff.
Partial post-hoc capture exists (stacker feature, logistic correction terms for size state), but not inside the Bayesian posterior.

**Proposal.** Hypothesis H = (pair (i,j), size lag k, state grouping g), direction d, ω on/off assignment, flip time s.
State h_t = 1[category(c_{t-k}) in group g]; gate bit u_t = w_{t-i} xor w_{t-j} xor 1[t >= s] xor ω.
Win: w_t | (u_t, h_t) ~ Bernoulli(ρ_{u h}), logit ρ_{u h} = a_u + γ_type · σ_{h xor d}, with σ = +1 for the larger-expected-size state,
a_u ~ existing on/off prior (with signal slope), γ_type the learned shared shift. Size: y_t | win, h_t ~ Cat(V_{type, h_t xor d}), V known.
Joint posterior: P(H,d,s,ω | w,y) ∝ P(H) P(s) · ∫ Π_u π(a_u) Π_h Bin(k_{uh}; n_{uh}, ρ_{uh}) da · Π_{t: w_t = 1} V_{h_t xor d}(y_t).
Closed-form version: give each sub-cell a Beta with mean shifted ±γ on the log-odds scale and the same strength (adi's `shift_cellprior`
already does this arithmetic). γ = 0 and the size factor off reproduce the present model exactly (nested). Prediction:
p(0) = 1 - E[ρ_{u_c h_c}], p(z) = E[ρ_{u_c h_c} · V_{h_c xor d}(z)] under the joint posterior (this also removes the product-of-marginals bias).
Learning: γ_type by train marginal likelihood on a grid (the same coordinate ascent as the signal slope), size factor on/off by train evidence.
Cost (estimate, not timed): learning +10-20% of the signal-slope step; prediction unchanged in order (7 structures, closed form);
tens of seconds to a few minutes on a laptop. Not implemented.

**Experiment (training only, negative control; not yet run).** On the stable train machines (192; condition labels select the
analysis subset only, never enter a model), split each history in half. From the FIRST half only: gate pair by MAP, orientation of on/off,
and each tick's size state from the existing size model. In the SECOND half, in on-cell ticks, compute D_m = logit(win rate | big-size state)
- logit(win rate | small-size state) (add-half smoothing). Prediction: mean D > 0 by about 0.5-0.9, sign agreement well above 55% in at
least two types. Controls: (N1) define the state from a decoy lag outside {i, j, k}; (N2) permute the first-half size-state labels across
machines of the same type. Both must give mean D inside +-0.15 and sign agreement near 50%. Falsified if the main D is inside +-0.15,
or either control matches it. If it passes, the keep-check protocol decides adoption.

## Entry 1: generator-constraint probes on main (tuning half only; diagnosis, no model change)

Tool: `python3 probes.py --data data/main` (48 tuning episodes, 384 queries; audit half not read). Nothing here is a model
change; each line is a hypothesis about how restricted the generator is, to be re-checked on the round-two kit.

| hypothesis (from the pasted threat list) | result on main tuning half | status |
| --- | --- | --- |
| Machine types share exact parameter constraints (sizes) | P(size given win) takes exactly 2 vectors per type: balanced [.500 .348 .130 .022] / [.214 .547 .205 .034], jackpot [.335 .245 .315 .105] / [.121 .324 .416 .139], steady [.455 .436 .107 .002] / [.754 .197 .048 .001]. 100% of queries on one of the two. | HOLDS. Already exploited by v8's size menu (learned from histories). Re-check on round two (day-one step 4). |
| Win probabilities come from a small discrete menu | True P(win) is continuous: 85-98% of values distinct in 8 of 9 type x signal cells (jackpot signal -1: 21 of 36, because probabilities near .003 collapse at 3 decimals). | FALSIFIED (agrees with Kareem's #34). |
| Signal encodes a hidden parameter more directly than we model | Signal does NOT predict the size state (hi-1x share 45-65% for every signal). Overall P(win) eta^2 for signal = .003. BUT inside a gate mode the logit of the win rate shifts monotonically with signal, about +0.3 to +0.55 logits per step in every type and mode. | PARTLY. The shift is real and already learned (signal slopes). A single shared slope across types is untested. |
| Hidden rate structure inside the on/off modes | Within-mode logit sd is nearly the same everywhere: balanced off .73 / on .65, jackpot off .72 / on .70, steady off .60 / on .53. | NEW LEAD (see below). |
| Episode-level sharing of rates | Episode eta^2 .120 vs chance level .125 (48 groups / 384). | No sharing (agrees with the ICC .07-.09 finding). |

**Lead from the within-mode spread.** Cell rates look like logit-normal: logit(rate) = base[type, mode] + beta * signal + N(0, sigma^2)
with sigma about 0.6-0.7 and beta about 0.4, close to shared across types. Our gate uses Beta priors per type with separate
per-type slopes. If the generator really is logit-normal with shared sigma and beta, a logit-normal prior (grid quadrature) with
shared sigma/beta, both learned in `fit()`, gives sharper priors on short logs. Cost: the gate's cell evidence is closed-form
Beta-Binomial, so this is a real code change in the gate. Prior art: mixture-of-Betas gave nothing (method4 #9), so expect a small gain.
**Tested (`probes.py --probe shape`, true cell rates on the tuning half, 18 type x mode x signal cells, 2 parameters each):**
logit-normal beats Beta by +5.8 log-likelihood units over 384 values (+0.015 per value); per cell the difference is -0.30
to +0.95. That is indistinguishable at this n, and these are the TRUE rates, a cleaner test than anything the gate sees.
**Verdict: do not rewrite the gate prior.** Beta is an adequate description of the cell rates. The only residual use of the
finding is the near-constant within-mode logit sd (~.6-.7) as a sanity check on the round-two kit (a changed sd would show up
in the fingerprint's cell-prior `k` values).

## Entry 0: baseline (v8 as copied)

- Hypothesis: none. This is the reference every later keep check compares against.
- Test: `python3 run.py score --name "v8 baseline"` on `../data/main`.
- Expected (from Kareem's log, e46c44d): tuning 58.309, audit 58.287, full 58.297. Verdict: reference.

## Entry 18: shared-base re-applied on Kareem's v9 (origin/main 0f201ad)
Copied v9 `method4_ml` (features3, model, size_menu, logit_stack, round2, polarity, test_no_leakage) into this folder; re-applied only the shared-base wiring
(`shared_base` option, `attach_shared_base` call in `_fit_gate`, `self.fver` cache-key suffix; off = pristine v9). Shared packages (adi/gate.py, adi/model.py, method2_sizes) unchanged on main, so our xsg additions apply as is. No teammate file touched.
Tuning half only (no audit use), official kit scorer, /60:
| run | tuning | stable | short | drift | shared |
|---|---|---|---|---|---|
| v9 patched tree, shared_base off | 58.3107 | .02035 | .05276 | .02723 | .01555 |
| v9 + shared-base | 58.3376 | .01854 | .05198 | .02827 | .01519 |
Identity: predictions of both runs equal the earlier v8 and v8+shared-base predictions exactly (max abs diff 0.0 over 768 keys), i.e. v9 at default settings reduces to v8 numerically and the shared-base gain (+0.0269 tuning) is reproduced. v9 learned types [steady, balanced, jackpot], short_obs 64, menu self-check keeps all three. Window protection (<=0.45 ceiling) still not implemented.
