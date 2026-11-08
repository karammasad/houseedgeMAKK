# Validation of the no-gate adaptation (pre-registered in PREREG_NOGATE.md before any run) -- accuracy passes, runtime check fails; NOT adopted by default
Adaptation (`nogate_fix.py`, wrapper in our folder; Kareem's Thanos 8e00a50 imported unmodified): add one no-gate candidate ((), size lag 1, initial mass .05) to the structure support.
## Test A: fresh synthetic kit (same frozen generator config, new seed 20261109, hidden truth; 1,536 test queries / 192 episodes), x1e-4 squared distance to the TRUE vector, T0 = frozen Thanos, T1 = with the fix
| subset | n | T0 | T1 | T0 - T1 [95% episode CI] | P(T1 better) |
| --- | --- | --- | --- | --- | --- |
| ALL | 1536 | 33.9 | 29.8 | +4.11 [+0.77, +7.56] | .991 |
| machines with a gate | 1441 | 25.5 | 23.3 | +2.29 [+1.24, +3.48] | 1.000 |
| no-gate machines | 95 | 160.5 | 128.8 | +31.7 [-19.0, +81.4] | .889 |
| no gate, balanced | 27 | 239.4 | 120.2 | +119.2 [+7.2, +255.8] | .983 |
| no gate, steady | 36 | 243.7 | 249.5 | -5.8 [-100.7, +75.8] | .465 |
| gated, stable / short_history / drift / shared_change | 357 / 370 / 364 / 350 | 11.3 / 69.7 / 9.9 / 9.7 | 9.9 / 63.6 / 9.5 / 8.5 | +1.4 / +6.1 / +0.4 / +1.2 | 1.0 / 1.0 / .89 / 1.0 |
The learned prior mass of the no-gate structure was .038 (truth about .05). The fix did not harm gated machines; it improved them (plausible mechanism, not verified: forcing no-gate machines through gated classes had been polluting the population parameters). The steady no-gate machines are NOT repaired: in my generator no-gate machines have no size state, whereas Thanos' size model applies one, so the residual may be a generator artifact (unverified).
## Test B: real Round 1 TRAIN, Thanos' own 4-fold episode-grouped CV (3,072 held-out rows, 96 episodes, no validation/audit)
Brier T0 3588.6, T1 3588.6; paired T1 - T0 = +0.00 [-0.00, +0.01] x1e-4 overall (every type and condition within +-0.05). Learned no-gate prior mass: about 3e-5 in every fold -> on real Round 1 data the fit switches the class off, so the adaptation is inert there (no harm, no gain). Pre-registered "no harm" bound (+3e-4): passed.
## Runtime
Fit time: Test A 1,044 s -> 1,611 s (+54%, both under heavy concurrent load); Test B per fold 212-279 s -> 437-524 s for the second fit (about 2x). The extra structure activates mu_n / ls_n in the per-type optimizer, which costs real fit time even when its mass goes to zero. Pre-registered check (iii) (overhead <= 25%) FAILS as measured; absolute times (about 7 min quiet, perhaps 11-15 min with the fix) stay far inside 30 min, but timings were taken under contention.
## Verdict
Pre-registered rule: all three checks required -> (i) pass, (ii) pass, (iii) fail -> the fix is NOT adopted by default. Evidence it is useful: helps on a synthetic kit with 5% no-gate machines (+4.1, P .99) and is inert on real Round 1 data. Recommendation: keep frozen Thanos E2 as the base; apply the fix ONLY if the Round 2 kit fingerprint shows no-gate / degenerate-rate machines (e.g. share of machines with win rate > .97 or < .03, or the fix's learned mass > ~1%); the cheaper alternative (route degenerate machines to the fallback) was not tested. Caveats: one synthetic generator that I wrote; concurrent-load timings; steady no-gate residual unexplained.
