# adi round two (adaptation): 9 PM runbook

Model: `adi.r2` = the 58.21 main model (`adi.sizewin`) with everything that was fit to main-round patterns widened and
re-learned from the new kit (details in `model.py`). On main data it scores tuning 58.238 (main model: 58.221), so
the widening costs nothing. Standard library only.

## At 9 PM (from `house-edge/`)

1. Download the round-two kit. Unzip it into its **own new folder**, outside the repo. Check its
   `public/metadata.json` says `"round": "adaptation"`.
2. Copy that kit's `public/` folder to `house-edge/data/adaptation/` (so `data/adaptation/train/queries.csv` exists).
   Don't touch `data/main/`.
3. Run:

   ```sh
   python3 -m adi.r2.run --data data/adaptation --out build/r2
   ```

   It fits two candidates (`r2` wide/adaptive, `main` = exact main settings), scores both on the new validation
   set, picks by the tuning half (audit guard), prints what was learned, writes and checks the test CSV, and
   builds the verified ZIP. Takes about 15-20 minutes on a laptop.
4. Upload **`build/r2/entry/submission.csv`** and **`build/r2/entry/entry.zip`** to the **Adaptation** form.
   Keep the confirmation.

## If something goes wrong

- Short on time: `python3 -m adi.r2.run --data data/adaptation --out build/r2_fast --only r2` (one candidate).
- The r2 model crashes: `--only main` (exact main-round model).
- Last resort, method 2's model (standard library):
  `python -m method2_sizes.make_entry --method method2_sizes --kwargs "$(python3 -c 'import json,method2_sizes.round2 as r; print(json.dumps(r.FINAL))')" --data data/adaptation --out build/r2_m2`
- Never upload a CSV whose ZIP wasn't built from the same code. The runbook's `entry/` folder pairs them.

## Rehearsal

Run on main data (prints a "not adaptation" warning, which is expected):
`python3 -m adi.r2.run --data data/main --out build/r2_rehearsal`. Results are in `adi/FINDINGS.md` ("Round two").
