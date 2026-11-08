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
2. **Fallback first** (plain v8, about 17 min): `python3 adapt.py --data ../data/adaptation --out build/r2_base --force base`
   -> `build/r2_base/submission.csv`, checked. Upload a fallback as soon as it exists (package it with step 5).
3. **Select** (about 3-4 min): `python3 adapt.py --data ../data/adaptation --out build/r2 --stage select`
   refits v8's gate, derives lag pairs from the fitted prior, fits the shared-base + size-shift structure, compares on the TUNING half (validation answers),
   one audit-half guard, writes `build/r2/chosen.json`. Prints derived pairs, gamma per type, convergence and bound hits.
4. **Final** (only if chosen = sb; about 17 min): `python3 adapt.py --data ../data/adaptation --out build/r2 --stage final`.
5. **Package** the chosen configuration: `python3 -m method4_ml.make_entry --data ../data/adaptation --out build/r2_entry --kwargs "$(python3 -c "import json;print(json.dumps(json.load(open('build/r2/chosen.json'))['kwargs']))")"`
   and check its CSV equals the one from step 2 or 4. Upload `submission.csv` + `entry.zip` to the Adaptation Form.
Tested: synthetic kits with changed lag pairs (adopts the structure) and with the size shift removed (falls back to v8); see FINDINGS Entry 10.

If scikit-learn is not allowed on the runner, the stdlib fallback is Adi's `adi.r2` runner (see `adi/r2/README.md` in the original repo).

## Log

- Copied from `e46c44d` on 2026-11-07. Leakage tests: see the run output in this session.
- Not yet run: full round-two rehearsal; packaging check from a fresh unzip.
