# Round 2 (adaptation) submission: Thanos + correlated on/off rate prior

**Recommended upload (matching pair):** `submission.csv` + `entry.zip` (both in this folder).
Fallback pair: `fallback_thanos_submission.csv` + `mominsreverseengineering/deploy_thanos/entry.zip` (plain Thanos, unmodified Round 1 code).

| file | sha256 |
| --- | --- |
| `entry.zip` | f9ce77747f296494339849a213b3e0f5aa48224733eee58fbde768e724ee4170 |
| `submission.csv` (1,536 test queries) | 3cdd5171760601f1003cf09f79ee472344c4a117a76a019ec0c82b2f09fa1919 |
| `fallback_thanos_submission.csv` | 71897dec75b5f6ba230aa75356fb894956933eb3375aa46bd112b76c3d535162 |

## What changed vs plain Thanos
Thanos puts INDEPENDENT normal priors on a machine's on-cell and off-cell win logits. Train histories show they are positively correlated
within a machine (Spearman of posterior means: balanced +.40 p 2e-8, jackpot +.22 p 2e-3, steady +.25 p 7e-4; Thanos' independent priors
cannot create this). The challenger uses a bivariate-normal prior, applied by exact Gaussian conditioning: mu_on's prior is shifted by
rho (s_on / s_off) r_off with variance s_on^2 (1 - rho^2) + (rho s_on / s_off)^2 v_off (r, v = the other side's own-history posterior mean /
variance under its independent prior), and vice versa; each side's own data enters once. rho per type from TRAIN histories only (method of
moments correcting for shrinkage; balanced .797, jackpot .757, steady .514). Code: `thanos/episode_effect.py` (`rho_hat`, `corr_predict`),
Thanos itself run with `adapt=False` (= Round 1 code path; byte-identical CSV to deploy_thanos/entry.zip verified locally).

## Validation (official metric, /40; validation.csv here = packaged run)
| | plain Thanos | challenger | diff |
| --- | --- | --- | --- |
| tuning | 39.061 | 39.075 | +0.014 |
| audit | 39.254 | 39.268 | +0.014 |
| full | 39.183 | 39.196 | +0.014 |
Paired episode bootstrap (full): error diff -0.000070 [-0.000144, +0.000002], P(better) .971, 54/96 episodes better, still better without
the top 5 episodes. Error ratio stable .0215 -> .0210, drift .0175 -> .0170, shared .0198 -> .0193, short .0246 -> .0248.

## Checks
- Leakage: `thanos/tests_leakage.py` on the submitted Thanos code, round-2 data: ALL 8 PASS. `thanos/tests_corr_prior.py` (new step): ALL 7 PASS.
- Runtime: fresh unzip, pinned numpy 2.4.4 / scipy 1.17.1, 2 threads: 663.6 s (11.1 min) wall, 198 MB peak (another job ran concurrently).
- Kit checker: VALID FORMAT, 1536 queries, adaptation, season 5fffdc4d0c700bbf61c3f911.
- UNVERIFIED: official Linux container; a second byte-identical rerun of the challenger package (only one pinned run).
- Overfitting caveat: chosen after three rejected ideas were scored on the same validation (DPCM, episode effect, discrete-menu test), so
  some selection bias is possible; evidence came from train first, nothing was fitted to validation.
