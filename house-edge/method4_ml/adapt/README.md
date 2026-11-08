# method4_ml/adapt: round-two engine (v9)

v9 is v8 with nothing from the main round hardcoded, plus a polarity bank that lets the stacker use rules the win gate cannot
express. A selector then chooses among v8 and v9 on the new kit's own validation data.

## Run it (from `house-edge/`)

1. Copy the round-two kit's `public/` folder to `data/adaptation/`. Check that `public/metadata.json` says `adaptation`.
2. Run:

   ```sh
   python3 -m method4_ml.adapt.run --data data/adaptation --out build/adapt
   ```

   This fits v8, v9 and v9 + polarity logistic term, scores each one on the new validation set, picks one, builds and
   verifies the ZIP, and checks that the ZIP's CSV matches the selection CSV.
3. Upload the CSV and ZIP from `build/adapt/entry/` (they come from the same build).

Options:
- `--quick` for a fast rehearsal (bags 2, seeds 2).
- `--skip-entry` to run selection only.
- `--wide` to also try the widened gate. It is slower, so measure the 2-core runtime before shipping it.

## What changed from v8

| Piece | File | What it does |
| --- | --- | --- |
| Learned machine types | `features3.type_order` | Main-round types keep their indices. A new type gets its own index instead of crashing. |
| Learned short-log shape | `model.learn_short_shape` | Main is "last 40 + 24 older" (reproduces v8 exactly). Any other shape copies real short-log masks. |
| Learned short threshold | `gm.short_obs` | Observed-tick count of a short log, taken from the data. |
| Polarity bank | `polarity.py` | For each machine, tests candidate bits on its own history: other machines' wins at lags 1-8, its own big-payout lags, and own XOR neighbour. Feeds the strongest evidence and the shift it predicts now to the stacker (12 features). |
| Polarity logistic term | `logit_stack.design`, spec `"pol"` | Structured offset-logistic version of the same signal. |
| Selector | `round2.py` | Default is v9 (trees + logistic). Another variant wins only if it is >= 0.05 tuning points better and the audit half is not worse. |
| Stress kits | `adapt/stress/make_kit.py` | Synthetic kits with known true probabilities: `main`, `cross`, `triple`, `sizes`. |

## Evidence (RESEARCH_LOG.md #40-43)

| Data | v8 | v9 | Note |
| --- | --- | --- | --- |
| Main validation (tuning / audit) | 58.309 / 58.287 | 58.305 / 58.287 | Same within noise; about +1% runtime |
| Cross-machine stress kit (full) | 34.95 | **39.84** | v9 + polarity logistic term; both halves agree |

- Leakage checks 1-4 and 1b (polarity features ignore every machine's ticks >= t) pass:
  `python3 -m method4_ml.test_no_leakage`.
- Falsified this round: a short/long-gated pool of v8 + adi.cook (#42) and gating by model disagreement (#43).
  v8 (58.297 full) remains the main-round champion.
- Stress kits are diagnostic evidence, not a forecast of round two.

## Known limits

- adi's cook model still hardcodes the 40 + 24 short mask (adi's code, not changed here).
- `make_entry` copies only top-level `.py` files of each package, so `adapt/` is not in the ZIP. It is not needed at
  prediction time.
