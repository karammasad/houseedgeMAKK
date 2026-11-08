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

## Round two (adaptation), once the kit is out

1. Unzip the new kit into its own folder. Check `public/metadata.json` says `"round": "adaptation"`.
2. Copy its `public/` to `../data/adaptation/`. Do not touch `../data/main/`.
3. `python3 -m method4_ml.round2 --data ../data/adaptation --out ../build/rev_r2 --no-cache` (reads FLAGS and the variant table).
4. Package: `python3 -m method4_ml.make_entry --data ../data/adaptation --out ../build/rev_adapt --kwargs "$(cat ../build/rev_r2/chosen.json)"`.
5. Check that the ZIP's CSV matches `../build/rev_r2/submission.csv`, then upload `submission.csv` and `entry.zip` to the Adaptation Form.

If scikit-learn is not allowed on the runner, the stdlib fallback is Adi's `adi.r2` runner (see `adi/r2/README.md` in the original repo).

## Log

- Copied from `e46c44d` on 2026-11-07. Leakage tests: see the run output in this session.
- Not yet run: full round-two rehearsal; packaging check from a fresh unzip.
