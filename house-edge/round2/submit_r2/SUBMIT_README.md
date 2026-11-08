# Round 2 (Adaptation) submission — ready to upload

**Files**: `submission.csv` (1,536 test queries) + `entry.zip` — upload both to the Adaptation form.

- `submission.csv` sha256: `e0228d7b273ba362caac4c4a96eee9947b1346ff462ea0969853685f590412d8`
- `entry.zip` sha256: `c74794a4cd07f58bea0f2e69017ce948b20a24b90387657650b0b3d915527513`

## What this is

thanos E2 (exact generator posterior), refit on the round-2 adaptation data (`data/adaptation`), transductive
(`--splits all --tickset own`), unchanged from the round-1 model — no code edits for round 2, it refits itself
from whatever `/data` points to.

## Validation score (official metric, max 20)

| | tuning | audit | full |
| --- | --- | --- | --- |
| thanos E2 | 19.5304 | 19.6271 | **19.5913** |

Per-condition (full, lower=better): stable 0.0215, short_history 0.0246, drift 0.0175, shared_change 0.0198 —
no localized weakness. Confirmed by **three independent runs**: Adi's original run, an independent recomputation
from his raw CSV, and a fully separate subagent refit from scratch — all match to 4 decimal places.

## Verification done

- Correct kit checker (`round2/kit_r2/check_submission.py`, NOT the stale `house-edge/kit/` copy which hardcodes
  an old 60/40 point split): `VALID FORMAT: 1536 queries | adaptation | test`, rc 0.
- `entry.zip`'s embedded `thanos/model.py`, `thanos/egp.py`, `common/data.py` are byte-identical (diffed) to the
  current `thanos/` source that both independent validation runs above actually used — so the packaged code is
  proven to be what produced these scores.
- Round-1 deployment audit (`mominsreverseengineering/deploy_thanos/VERIFY.md`) already verified this exact zip
  builds with the kit's packager, unzips clean, runs offline (network denied), reads only queries.csv+history.csv,
  writes only the output CSV, and reproduces byte-identically on reruns in the same environment.

## NOT done (known gap, accepted given the 90-minute window)

A literal "fresh unzip entry.zip, run predict.py --data data/adaptation from scratch, diff the resulting CSV
against this submission.csv" pass was not completed before upload — skipped to preserve time buffer before the
12:00 PM deadline, given the strong substitute evidence above (byte-identical source, 2x independently matching
full-pipeline scores). If time permits after upload, run:
```
cd /tmp && rm -rf thanos_r2_final_verify && mkdir thanos_r2_final_verify && cd thanos_r2_final_verify
unzip -o -q /Users/karammasad/houseedgeMAKK/round2-live/house-edge/round2/submit_r2/entry.zip
python3 predict.py --data /Users/karammasad/houseedgeMAKK/round2-live/house-edge/data/adaptation --out /tmp/final_check.csv
diff <(sort /tmp/final_check.csv) <(sort /Users/karammasad/houseedgeMAKK/round2-live/house-edge/round2/submit_r2/submission.csv)
```
Empty diff = fully confirmed. Non-empty (expected small float noise from parallel BLAS across separate runs,
same pattern already seen between our two prior runs' differing sha256s but identical scores) does not invalidate
the submission — the model and score are what matter, not exact test-CSV byte-reproduction under threaded fp noise.

## Other candidates considered (not chosen)

- adi.r2 (stdlib, no sklearn risk): 19.430 full — clearly behind thanos, kept as a safety net if thanos turns out
  ineligible/broken for some unforeseen packaging reason.
- method4_ml v9+shared-base, thanos+no-gate-fix: still running at upload time; see round2/README.md "Status log"
  for whether either finished with something better before the deadline.
