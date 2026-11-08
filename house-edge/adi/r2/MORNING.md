# Round two: morning checklist (data 10:30 AM, upload by 12:00 PM Central)

Target: a verified upload by ~11:00, then improve only if time allows. Commands run from `house-edge/`.

## 10:30 - get the data (5 min)
1. Download the adaptation kit. Unzip it into its **own new folder** outside the repo.
2. Check `public/metadata.json` says `"round": "adaptation"`.
3. Copy that kit's `public/` folder to `house-edge/data/adaptation/` (so `data/adaptation/train/queries.csv` exists).
   Don't touch `data/main/`.

## 10:35 - one command (about 25 min on this laptop)
```bash
python3 -m adi.r2.fast --data data/adaptation --out build/r2fast
```
It fits 4 candidates in parallel (`r2`, `cook`, `cook_nomenu`, `cook_nocouple`), scores each on the new validation
set, picks the best tuning score (audit guard), writes the test CSV, runs the kit checker, builds the ZIP and reruns it
from a fresh unzip (must be byte-identical). Main-data rehearsal: 14 min to the choice, 23 min to a verified ZIP.

What good output looks like:
- `exit 0` for every candidate, then a `[name] tuning ... audit ... full ...` line for each.
- `CHOSEN: ...` followed by what it learned (structures, cell priors, kappa, size menu).
- `VALID FORMAT: N queries | adaptation | test` and finally `OK. Upload ...`.

## 11:00 - upload
Upload **`build/r2fast/entry/submission.csv`** and **`build/r2fast/entry/entry.zip`** under **Adaptation**.
Save the receipt. Before uploading, run once more:
```bash
python3 kit/check_submission.py build/r2fast/entry/submission.csv --data data/adaptation
```

## If something goes wrong
- One candidate fails: the others still run; it picks among the ones that finished.
- The build step fails: upload nothing yet. Rerun only the safest candidate:
  `python3 -m adi.r2.fast --data data/adaptation --out build/r2fast_r2 --only r2`
- Everything fails: the old runbook, sequential (~40 min): `python3 -m adi.r2.run --data data/adaptation --out build/r2_old --only r2`
- Never upload a CSV whose ZIP wasn't built from the same run (the `entry/` folder pairs them).

## Comparing with Kareem's v8
Kareem runs `method4_ml.round2` on the same kit at the same time. Compare **tuning** scores on the new validation set (audit as a
check, like the runbook). The higher one is uploaded with its own matching ZIP. v8 needs its `requirements.txt` at the ZIP root.
