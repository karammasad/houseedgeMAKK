# Round-two preparation: your shared-base model

**Experimental research only.** The selector and workflow have not been independently validated end to end on both generator families. Do not treat them as a production submission or an automatically approved model-selection procedure.

## What changes

The organizer kit says adaptation changes which temporal relationships recur. It provides new train, validation and test histories; validation includes true probabilities and the round-specific frozen reference. The score is out of 40 and weights stable, short history, drift and shared change equally. The published kit says 14 hours, but this workflow targets a 90-minute working window. Check the actual organizer announcement for the release and cutoff; teammate MORNING.md gives a different schedule from the kit.

Use only ticks strictly before each predicted tick; missing ticks are unknown. Never give validation/test targets, labels, conditions or reference files to fit/predict. Main-round data may be reused, but changed relationships make its usefulness uncertain.

## What was learned from the team

- Your shared-base gate derives its candidate pairs and population parameters from the new train histories. It remains constrained to the existing gate family and a size-state split; check its fit diagnostics on the new data.
- Kareem's existing polarity feature bank is already present in this folder's v9 snapshot. It tests cross-machine and category signals. On a synthetic cross-machine kit his logged model improved, but that is not evidence it will improve this kit. We test your shared-base model with the bank and `pol` logistic term as a second, fixed candidate.
- The two-vector size menu has a per-type, history-only self-check. Read the fit output to see whether it stays enabled. Do not assume the main-round menu still holds.
- Adi's hierarchical cook and the 50/50 pool produced 58.3689 on main, but the main-round checkpoint cannot be used as a round-two fit. His fresh-synthetic check did not independently establish that shared-base improves the pool. The current script **does not fit or package cook or a pool**. If the team independently runs a new-round cook, compare it using the new tuning data; pooling requires separately fitted checkpoints and a separately verified two-process entry. Do not submit the old main-round ZIP for adaptation.

## Before data arrives

From this folder (`house-edge/mominsreverseengineering`), verify Python dependencies are installed. The builder pins the installed versions into the entry; do not change versions between fitting and packaging. Current local versions differ from the old requirements file, so verify the runner can install the versions actually used. Keep enough time for fresh-unzip verification. The organizers' separate rerun limit is 30 minutes, 2 cores, 16 GiB.

## When the kit arrives

1. Put the new kit under `house-edge/data/adaptation/` without replacing main. Check metadata and dependencies without reading validation answers:

   ```sh
   python3 round2_pool/preflight.py --data /absolute/path/to/adaptation/public
   ```

2. Start the workflow with a new absolute output path:

   ```sh
   bash round2_pool/run.sh /absolute/path/to/adaptation/public /absolute/path/to/new-output 4
   ```

   It fits the shared-base fallback on the new data, generates and checks a checkpoint entry, and prints the matching CSV/ZIP paths as soon as the fresh-unzip verification finishes. Upload that fallback early if the deadline is close. It then fits one pre-specified polarity candidate and compares frozen validation forecasts.

3. The candidate must beat the fallback by at least 0.15 tuning points on a 60-point scale (0.10 on a 40-point kit), with no tuning condition more than 10% worse. Only then does the selector open the audit for one comparison. It additionally requires the same scaled gain on audit and no audit condition worse. This combines the repo's tuning rule with the more conservative audit rule in CONTROL.md. An audit sample this small can still be noisy. Do not rerun selection with variants after inspecting audit.

4. If selected, the candidate is packaged and checked independently; otherwise the verified fallback remains the answer. The script prints the correct paired files. Use only matching CSV and ZIP from the same entry folder. Submit through the adaptation form and retain confirmation.

The 90-minute budget is an operating target, not a tested guarantee for a new kit or a different machine. A fresh shared-base fit took about 17 minutes on 8 cores in the main-round experiments, and its 2-core cold fit took 27.2 minutes; packaging uses checkpoints. The extra polarity fit and packaging add time. If the fallback is still running near the deadline, prioritize its completion and submission over the candidate. The script never uploads.

## Rehearsal and limitations

`preflight.py --allow-rehearsal` accepts main or synthetic kits. `fit.py --allow-rehearsal` similarly permits other rounds. `compare.py --allow-main` permits a main-round rehearsal, but that audit is already consumed. These flags do not make a synthetic score predictive of round two.

Synthetic changed-lag drill: the shared-base + polarity candidate finished fit and both predictions in 406 seconds on this machine while another process was running. Its tuning was 39.3358 versus 39.3739 for the saved shared-base baseline, so the selector retained the baseline without scoring audit. The candidate's 160-row test CSV passed the kit checker. This synthetic drill did not exercise the full ZIP-building workflow and does not establish a 90-minute completion guarantee.

The script tests only shared-base versus shared-base + polarity; it does not establish that either is better than plain v9 or the cook pool. `adapt.py --stage full-select` compares plain versus shared-base, but its current full-select implementation prints only the chosen model's audit score and does not veto against the baseline; use this folder's `compare.py` for an actual fixed-candidate audit comparison. If runtime or data changes make the automated candidate impractical, ship the verified fallback and keep its exact files.

## Thanos challenger

After this workflow was drafted, Kareem added Thanos (Git revision `360ebc6`). See `THANOS_REVIEW.md` for the independent main-round I/O audit, timing, synthetic changed-lag check, and remaining limitations. The current runner does not fit or package Thanos. Its higher main validation score does not establish round-two performance; compare freshly refitted candidates on the new tuning split and verify an entry's matching code ZIP before submitting.
