# round2: adaptation-round orchestration (karam)

Prepared ahead of the kit drop (round two opens 10:30 AM, closes 12:00 PM Central, 90 minutes). This folder
only reads other teams' folders (`thanos/`, `method4_ml/`, `adi/`, `mominsreverseengineering/`); nothing in
them is edited. Working tree: `round2-live` worktree, branch `round2-live` off `origin/main` @ ea93475.

## Where things stand (round 1)

- **thanos** (exact generator posterior) is the submitted round-1 champion: 58.629 full validation vs the
  previous champion method4_ml v9+shared-base at 58.355. Code: `thanos/`, deploy copy `mominsreverseengineering/deploy_thanos/`.
- Momin's **synthetic round-2 rehearsal** (`mominsreverseengineering/experiments/rehearsal/RESULT.md`): frozen
  thanos E2 fails the team's pre-registered adopt rule on synthetic round-2-like data (P(better)=.895 < .95,
  drift 14.8% worse). Root cause isolated: thanos has **zero support for no-gate machines** (always-win /
  always-lose, ~5-6% of machines) — outside those, thanos wins by about a third. A monkeypatch fix
  (`mominsreverseengineering/experiments/rehearsal/nogate_fix.py`) recovers this on synthetic data (+4.1, P .991)
  and is inert (no harm) on real round-1 train CV, but costs +54-100% fit time (absolute time still well inside
  the 30-min cap). **Not adopted by default** — apply only if the real fingerprint/probes show no-gate or
  degenerate-rate (win rate >.97 or <.03) machines.
- Fallback: **method4_ml v9+shared-base**, with an existing round-two runbook (`mominsreverseengineering/adapt.py
  --stage full-select`) that fits plain v8/v9 and the shared-base candidate as two full pipelines in parallel,
  selects on the official tuning-half metric (candidate must beat base by >=0.05 pts on a 60-pt scale, worst
  condition not >10% worse), then checks the audit half once as a guard.
- Safety net (stdlib only, no sklearn risk): **adi.r2** (`adi/r2/run.py`), ~58.2 full on main, fits a wide/adaptive
  candidate + the exact main-round model, picks on tuning (audit guard).

## IMPORTANT: use round2/kit_r2/check_submission.py, not kit/check_submission.py

The repo's `house-edge/kit/` folder is a stale copy from the main-round kit download (day 1). Its
`slot_forecasting/rounds.py` hardcodes `ROUND_POINTS = {"main": 60, "adaptation": 40}` and will raise
`ValidationError: Expected main=60 or adaptation=40 in metadata; old 80/20 kits must be updated by
organizers.` against the real adaptation kit, whose `metadata.json` correctly says `max_points: 20`
(round="adaptation") — matching the participant guide ("Main: 80 points. Round two: 20 points.").
The freshly-downloaded adaptation kit ships its own correct checker (`ROUND_POINTS = {"main": 80,
"adaptation": 20}`), copied here unmodified as `round2/kit_r2/`. **Use this for every round-2 format
check and validation score**, e.g.:
```
python3 round2/kit_r2/check_submission.py <csv> --data data/adaptation
python3 round2/kit_r2/check_submission.py <validation.csv> --data data/adaptation --split validation
```
(`house-edge/kit/` is left untouched — it's still correct for the closed main round, whose
`data/main/metadata.json` says `max_points: 60`. Don't overwrite it.)

## Day-one checklist (do this the moment the kit lands)

1. Unzip the new kit into its own folder (outside the repo). Confirm `public/metadata.json` says `"round": "adaptation"`.
2. Copy `public/` to `house-edge/data/adaptation/` (never touch `data/main/`). `round2/setup_kit.sh <unzipped_public_dir>` does this.
3. Fingerprint + diff (safe: train/history only, no targets):
   ```
   python3 mominsreverseengineering/fingerprint.py --data data/adaptation --out round2/results/fp_adapt.json
   python3 mominsreverseengineering/fingerprint.py --diff round2/results/fp_main.json round2/results/fp_adapt.json
   ```
   `round2/results/fp_main.json` is pre-computed (main-round baseline, see below) so the diff is instant.
4. Probes on the tuning half only: `python3 mominsreverseengineering/probes.py --data data/adaptation`
   Check in particular: does a no-gate / degenerate-rate population show up? That decides whether to run the
   thanos no-gate-fix variant (job A2 below).

## The three (four) parallel jobs

Run each as its own subagent once data/adaptation exists. All commands run from `house-edge/` (this worktree root).

**Job A — thanos, frozen E2, refit on new data** (~7-13 min per VERIFY.md / rehearsal timing):
```
python3 -m thanos.run --data data/adaptation --splits all --out round2/results/thanos_r2 --also validation
python3 round2/score_pred.py --data data/adaptation --pkl round2/results/thanos_r2/validation.pkl --name thanos --out round2/results/thanos_r2_score.json
```

**Job A2 — thanos + no-gate fix** (ONLY if step 4's probes/fingerprint show a no-gate or degenerate-rate population):
```
python3 round2/run_thanos_nogate.py --data data/adaptation --out round2/results/thanos_nogate_r2
python3 round2/score_pred.py --data data/adaptation --pkl round2/results/thanos_nogate_r2/validation.pkl --name thanos_nogate --out round2/results/thanos_nogate_r2_score.json
```

**Job B — method4_ml v9+shared-base, official round-two runbook** (~20-30 min, runs base+sb concurrently):
```
cd mominsreverseengineering
python3 adapt.py --data ../data/adaptation --out ../round2/results/r2 --stage full-select --workers 4
cd ..
```
Reads `round2/results/r2/chosen.json` (which candidate + kwargs) and `round2/results/r2/{base,sb}/final.json` (tuning/audit/full points for each).

**Job C — adi.r2 stdlib safety net** (~15-20 min):
```
python3 -m adi.r2.run --data data/adaptation --out round2/results/adi_r2
```
Reads `round2/results/adi_r2/...` for its chosen candidate + scores (see `adi/r2/README.md`/`MORNING.md` for exact output layout — check on day).

## Decision rule (team's existing bar, applied across all candidates)

1. Compare full/tuning/audit points across whichever of A, A2, B, C finished and passed the kit checker.
2. Prefer thanos (A or A2) unless: it fails checker/runtime, or B/C beats it by the team's adoption bar
   (>=0.15 tuning points, audit moves the same way, no condition >10% worse) — matches `AGENTS.md` rule 6
   and `mominsreverseengineering/CONTROL.md`.
3. If A2 was run: adopt only if it beats A on the SAME new-kit tuning half by a real margin and audit does not
   get worse (mirrors the synthetic-kit test protocol in RESULT_NOGATE.md).
4. If nothing finishes in time / everything errors: fall back to C (adi.r2, stdlib, fastest to trust, no sklearn
   version risk) or B's `base` (plain v8/v9, no shared-base) — both have their own internal fallback-first logic.
5. Package the winner, run `kit/check_submission.py`, and verify entry.zip code matches the CSV before handing
   off for upload.

## Packaging the winner

- **thanos**: the round-1 deploy package `mominsreverseengineering/deploy_thanos/entry.zip` is DATA-PATH AGNOSTIC
  (`predict.py --data /data --out /output/submission.csv`, fits fresh every run) — the *same unmodified zip*
  works for round two. Just regenerate `submission.csv` by running it against `data/adaptation` and reuse the
  existing `entry.zip` as-is. No repackaging needed unless the no-gate fix is adopted (then package a small
  variant that imports `nogate_fix` — see `round2/run_thanos_nogate.py` once written).
- **method4_ml (B)**: `python3 -m method4_ml.make_entry --data data/adaptation --out round2/results/r2_entry --kwargs "$(python3 -c "import json;print(json.dumps(json.load(open('round2/results/r2/chosen.json'))['kwargs']))")"`
- **adi.r2 (C)**: packaging is built into `adi.r2.run` (see its README) — check `round2/results/adi_r2/entry/`.

## Status log

- 10:26 CST: kit not yet released. Worktree `round2-live` created off fresh `origin/main` (ea93475). `data/main`
  symlinked in (read-only rehearsal data). `round2/score_pred.py` written (generic validation scorer for any
  pickled prediction dict). Main-round fingerprint baseline running -> `round2/results/fp_main.json`.
- 11:30 CST: FINAL RESULT. All four candidates finished, scoreboard below (official metric, max_points=20).
  Adi had already submitted plain thanos (19.5913 full) before the other candidates landed. Nothing beat it;
  the no-gate-fix variant is statistically tied (+0.0002, pure noise, far below the adoption bar) and everything
  else trails by 0.11-0.26 points. **No replacement submitted. Plain thanos stands as the team's adaptation
  entry.**

  | candidate | tuning | audit | full |
  | --- | --- | --- | --- |
  | thanos (plain E2, submitted) | 19.5304 | 19.6271 | **19.5913** |
  | thanos + no-gate fix | 19.5305 | 19.6274 | 19.5915 (tied within noise, not adopted) |
  | method4_ml sb (v9+shared-base, chosen by its own internal rule over base) | 19.3683 | 19.5591 | 19.4795 |
  | adi.r2 (stdlib) | 19.3340 | 19.4940 | 19.4300 |
  | method4_ml base (v9) | 19.3444 | n/a | n/a |

  Why thanos wins decisively: it's the only candidate using the full exact-generator-posterior approach (joint
  probabilistic model per machine, every hyperparameter fit by marginal likelihood on histories only); the
  others are gate+tree-stacker or frequency-based approaches with less structural fidelity to this generator.
  The round-1 champion's margin over the field (which was much smaller, ~58.6 vs ~58.3-58.4 out of 60) widened
  substantially on round 2's data (19.59 vs ~19.3-19.4 out of 20) -- consistent with thanos's design generalizing
  well to a kit with different (re-estimated, not hardcoded) lag structures.
