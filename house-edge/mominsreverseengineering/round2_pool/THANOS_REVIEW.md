# Independent review of Thanos (git revision 360ebc6)

## Strategy

Kareem fits a joint empirical-Bayes model to pre-cutoff payout histories. It averages over own-machine XOR lag candidates, payout-state direction, a shared relationship between win and payout states, polarity, and a change point, with rate uncertainty integrated on a numerical grid. It jointly learns the payout-size vectors, rather than using our separate gate, size model, stacker, and projection. He screens structures, then fits only the retained support. This is an architectural challenger, not our current 50/50 pool.

The `own` variant uses a per-structure observed-tick set and plugs in empirical marginal probabilities for observations that lack required lag values. Those empirical probabilities are estimated from the same history being scored. This is an approximation, not an exact integration over missing history and rate uncertainty. It is history-only, not an observed query answer leak. It needs train-only validation.

## Independent run

On main data, fresh `splits=all, tickset=own` execution loaded directly from the Git object database through our read-only importer. Only train/validation/test `queries.csv` and `history.csv` were opened during fit and prediction; no answer, target, condition, or reference CSV was opened. Both validation and test produced finite, nonnegative 5-vectors summing to one for all query IDs. Official test checker: VALID, 1,536 queries.

- Thanos: tuning 58.493704, audit 58.785425, full 58.629303 / 60.
- Our saved shared-base/cook pool: tuning 58.361422, audit 58.382051, full 58.368919 / 60.
- Full gain +0.260384. All four full-validation condition ratios improved: stable 14.2%, short 19.3%, drift 9.1%, shared 18.3%.
- Paired bootstrap over episodes within condition, 5,000 draws, seed 917: interval [+0.036, +0.468], P(gain > 0) 0.989. This describes sampling variation conditional on the selected model; it is not protection against selecting the model after looking at validation.
- 2 BLAS threads, one process: fit + validation + test wall 388.8 s, CPU 363.4 s, peak RSS 186.5 MB; fit alone 345.2 s. This was a shared local machine, not the organizers' container. No complete ZIP / offline fresh-unzip reproduction for Thanos was checked here.
- Exact open-file trace and timings: `build/thanos_independent_audit/audit.json` (local, ignored). Test CSV: same folder. Audit harness: `round2_pool/audit_thanos.py` (executes pinned Git content, does not copy teammate code).

## Limits and round-two risks

The `own` tick-set variant was chosen after seeing results for the original variant. Main validation and its audit have been repeatedly inspected; their gains are not an untouched confirmation. The checked-in log reports episode-grouped CV for the original `common` variant, and explicitly says the decisive `own`-versus-`common` train CV and final-variant leakage tests are pending. Our file-open audit independently covers a full final-variant run, but is not equivalent to all future-invariance and stripped-data tests.

The structure search covers own-machine gates with up to two win lags, and lag support is pruned at 1% screening mass. It learns two size vectors per type, and its population fit uses an early long-log prefix or the whole short log as approximately flip-free. A new generator may violate these assumptions; the independent validation kit must decide. The input lag range of up to 8 matches the published guide, but that does not guarantee the functional family is complete. The main audit evidence cannot guarantee adaptation generalization.

Use the measured runtime as encouraging, not a full deployment certification: the revision contains a run script but no verified self-contained entry ZIP and fresh-folder comparison in its `thanos_method` directory. Package with exact fitted dependencies and verify offline reproduction and runtime under the actual 2-core runner before submission.

## What we can explore in our own model

The most specific lead is that joint size/win evidence informs the shared latent direction and rule. Our earlier win-informed direction hook was independently derived and tested on training only (FINDINGS Entry 15); its short-log benefit was too small and uncertain, so it remains disabled. Thanos provides a reason to re-examine joint evidence, not evidence that switching that hook on improves our model. A safe next experiment is to predefine a train-episode-held-out comparison of coupled versus uncoupled direction evidence, include controls and measure all five-category loss; retain the frozen model unless it improves on fresh data. Do not copy Thanos code into this folder or reuse main validation as the final promotion test.

## Changed-lag synthetic check

After isolating our work in a separate clone, fitted the unchanged revision to the existing synthetic adaptation kit (`experiments/synth_changed/public`). Its three leading gate pairs were the kit's three generated pairs. Thanos scored tuning 39.36964, audit 39.62865 and full 39.48114 / 40. The previously saved shared-base fit scored 39.37387, 39.10868, 39.28670 respectively. The synthetic audit had already been used by earlier experiments; this is a diagnostic, not a clean promotion test. Thanos's test CSV was valid for all 160 synthetic queries. Fit + both predictions took 274.3 s, with two BLAS threads and ~111 MB peak RSS, in this clone. One synthetic generator cannot establish round-two robustness.

A fixed 50/50 blend of the main-round Thanos predictions and our saved shared-base/cook pool gave tuning 58.5077, audit 58.6796, full 58.5870 versus Thanos alone 58.4937, 58.7854, 58.6293. It is exploratory on consumed validation and was not adopted.
