# Mini Thanos cross-machine stress — discussion draft

Source revision: `8e00a50` (Mini Thanos); its Thanos E2 implementation is unchanged from `360ebc6`. The simulation and reporting scripts here are our own, in the isolated clone. No production files, official validation, or official audit were used. No teammate code was modified.

Predefined stress: every machine's win gate is driven by its neighbor's win at lag 3. The kit allows earlier neighbor outcomes to help, but does not say this particular rule occurs in round 2. This is one deliberately extreme out-of-family scenario, not a sample from the real simulator. The generator uses its own population rates and size vectors, so it does not isolate the effect of the rule from every possible generator difference in a paired control.

Separate synthetic training and held-out validation episodes: 32 each, with 8 episodes per condition and 8 machines per episode. Predictors were fixed before viewing held-out truth. Thanos fitted on permitted train/validation/test histories; shared-base and shared-base + polarity fitted normally. Mini is assembled exactly as the published route (short -> shared-base, long -> Thanos) using those fitted predictions. Every candidate saw the same histories and held-out true probabilities.

Five-class squared error averaged within episode then equally over the four conditions (lower is better):
- Mini 0.165596
- cross-machine shared-base + polarity, with fixed pol logistic term 0.083985
- Thanos E2 0.161633
- shared-base 0.165613

Paired, condition-stratified episode bootstrap, 5,000 draws, 32 held-out episodes (95% intervals):
- Mini minus polarity +0.081611 [0.071164, 0.092415]
- Mini minus Thanos +0.003963 [0.000199, 0.007555]
- Mini minus shared-base -0.000017 [-0.005079, +0.004891]
- shared-base minus polarity +0.081628 [0.071753, 0.091043]

On held-out win probabilities, mean squared error was 0.11545 for shared-base and 0.05828 for polarity (Thanos 0.11309); polarity win probability correlated 0.810 with truth. This demonstrates useful recovery of the cross-machine signal under this strong synthetic rule, not exact recovery or proof of benefit on a real kit.

Before opening held-out truth, a fixed train-only diagnostic compared sampled training-cutoff outcomes at the episode level. Polarity-minus-Mini Brier was -0.11023, episode-bootstrap CI [-0.13105, -0.08925] across 32 episodes. The pre-set rule (choose polarity only when the CI is entirely favorable) selected polarity. This prototype compares fitted candidates using permitted training labels, with no query label entering either fit. Its selection policy and uncertainty need validation across further unseen generators before promotion; it is not a universal fallback guarantee.

Timing: dataset generation 07:03:50 Central, fixed train decision 07:24:06, approximately 20m16s total; shared-base fit + synthetic validation + test 526.0s, polarity fit + both predictions 574.9s (2 workers, 1 BLAS thread), Thanos fit + train/validation/test 56.8s (one process, 2 BLAS threads). Mini's synthetic components took about 583s sequentially including a full shared-base fit; published checkpoint-based mini run reports 564s for fit/predict on main. These are local wall times, not certified 2-core container runs; the 2-core 30-minute organizer limit includes offline reproduction. Mini lacks a verified self-contained ZIP; no fresh-unzip timing has been measured. The Thanos ZIP is also not verified here. Its clean packaging and offline verification require further work; no new fit was started after this test.

Recommendation: Mini is a strong conservative candidate for familiar own-lag mechanisms, not a robust default under an unknown neighbor-driven change. Refit candidates on round-two data; compute a pre-set episode-held-out or truly held-out training-cutoff diagnostic for cross-machine gain and retain a separately validated, packaged polarity-enabled fallback when evidence supports it. Do not use this stress kit to claim competition ranking or choose a new threshold after seeing its result. No candidate was promoted, merged, or pushed.
