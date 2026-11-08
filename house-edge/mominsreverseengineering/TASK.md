# TASK: reverse-engineer and adapt the v8 model (owner: Momin)

Read CONTROL.md first. Starting point: `method4_ml` v8 (Kareem, commit e46c44d), copied here. Its FINAL is in `final.json`.

## Queue (do in order, one hypothesis at a time)

1. **Baseline.** `python3 run.py score --name "v8 baseline"`. Must match 58.309 / 58.287 / 58.297 (kit scoring). Log it.
2. **Leakage.** `python3 -m method4_ml.test_no_leakage` passes. (Done once on the copy: rerun after any change.)
3. **Round-two sensitivity.** Which parts of v8 are fit to main-round patterns? For each learned piece (structures,
   signal shift, size menu, stacker), note whether it re-learns from the new kit. Write findings to FINDINGS.md.
4. **Gate family.** Does the wider adi.r2 structure family (`adi/r2`) match or beat the main gate on main data?
   Keep check against FINAL. Do not swap it into FINAL unless KEEP.
5. **Stacker blend.** Check the blend weight and the logistic correction on the round-two rehearsal. Keep the simpler
   model if the stacker stops helping (round2.py MARGIN rule).
6. **Packaging.** Fresh-unzip rerun must reproduce the CSV byte for byte. Requires the pinned scikit-learn stack.

## Status

- 2026-11-07: folder created from e46c44d. Workflow files added (CONTROL.md, run.py, FINDINGS.md). Baseline not yet logged.
