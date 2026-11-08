# mominsreverseengineering

My working folder for reverse-engineering and round-two work on the strongest model (Kareem's method4_ml v8).

## What this is

A **snapshot copy** of the code v8 needs, taken from `house-edge/` at commit `e46c44d` (origin/main):

| folder | copied from | why |
| --- | --- | --- |
| `method4_ml/` | `method4_ml/` (Kareem) | the v8 model (`model.py`), size menu, stackers, round-2 runner, packager |
| `adi/` | `adi/` (Adi) | win gate (`adi.model`), size-split gate, `adi.stack` trees |
| `method2_sizes/` | `method2_sizes/` (Momin) | `GateSizes` size model that v8 imports |
| `method1_gate/` | `method1_gate/` | imported by some shared code paths |
| `common/` | `common/` | data loader and scoring (`common.evaluate`) |
| `requirements.txt` | `method4_ml/requirements.txt` | pinned scikit-learn stack |

Excluded on purpose: `cache/` (rebuilt on run), `out/`, `analysis/` scripts, and the other teammates' method folders.

**Nothing in the original folders was edited.** This folder is independent: if a teammate changes their code, our copy does not change unless we re-copy it on purpose. Original data stays in `house-edge/data/` and is passed in with `--data`, never copied here.

## Quick start for teammates (v8 + optional shared-base structure)

```sh
git fetch origin && git checkout mominsreverseengineering && cd house-edge/mominsreverseengineering
python3 -m pip install -r requirements.txt            # needs house-edge/data/main/ (the kit's public/ folder, gitignored)
python3 run.py score --name "try" --kwargs '{"shared_base": true}' --tuning-only   # candidate; omit --kwargs for plain v8; ~17 min fresh, 30 s with caches
```
Status: INCONCLUSIVE as a Round 1 upgrade (+0.057 full validation, see FINDINGS Entry 12); plain v8 stays primary. The audit half is consumed: do not tune on it.

## Setup

```sh
cd house-edge/mominsreverseengineering
python3 -m pip install -r requirements.txt   # use the pinned versions; see the note below
```

Note: the local machine currently has scikit-learn 1.8.0, but `requirements.txt` pins 1.9.1. Install the pinned versions before building anything that will be uploaded.

## Checks

```sh
python3 -m method4_ml.test_no_leakage        # leakage tests (passed on the copy, see log)
```

## Scoring (validation)

```sh
python3 -m method4_ml.round2 --data ../data/main --out ../build/rev_main --no-cache
```

Run the round-two script on main data first as a rehearsal. Before 9 PM Saturday there is no adaptation data; once it arrives, put it in `../data/adaptation/` and use `--data ../data/adaptation`.

## Round two (adaptation), once the kit is out (90 minutes)

1. Unzip the new kit into its own folder; copy its `public/` to `../data/adaptation/` (never over `../data/main/`).
2. **One command, fallback included** (about 20-30 min on 8 cores): `python3 adapt.py --data ../data/adaptation --out build/r2 --stage full-select`
   runs plain v8 (`build/r2/base/`, the fallback, with its own checked CSV) and v8 + shared-base structure (`build/r2/sb/`) as two FULL pipelines in
   parallel, chooses on the OFFICIAL tuning-half metric (candidate must gain >= 0.05 points on a 60-point scale, scaled to the kit, and no condition
   worse than 10%), then looks at audit once for the choice, and copies the chosen CSV to `build/r2/submission.csv`.
   If time is short, upload `build/r2/base/submission.csv` the moment it exists (package it with step 5, kwargs `{}`).
3. Optional diagnosis (3-4 min): `python3 adapt.py --data ../data/adaptation --out build/r2_diag --stage select --no-guard` prints the derived pairs,
   gamma per type, convergence and bound hits. Do NOT use its gate-level choice to decide: on main it chose the structure (-9% win-side) while the
   full pipeline gained only +0.027 tuning points.
4. A kit-checker problem is reported in the log and in final.json (`checker_ok`); it never discards a finished fit.
5. **Package** the chosen configuration: `python3 -m method4_ml.make_entry --data ../data/adaptation --out build/r2_entry --kwargs "$(python3 -c "import json;print(json.dumps(json.load(open('build/r2/chosen.json'))['kwargs']))")"`
   and check its CSV equals the one from step 2 or 4. Upload `submission.csv` + `entry.zip` to the Adaptation Form.
Tested: synthetic kits with changed lag pairs (adopts the structure) and with the size shift removed (falls back to v8); see FINDINGS Entry 10.

If scikit-learn is not allowed on the runner, the stdlib fallback is Adi's `adi.r2` runner (see `adi/r2/README.md` in the original repo).

## Log

- Copied from `e46c44d` on 2026-11-07. Leakage tests: see the run output in this session.
- Not yet run: full round-two rehearsal; packaging check from a fresh unzip.
