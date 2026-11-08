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

## Round-one reuse candidates

`round2.py` also fits these when round-one data is available. It looks for `--main`, then `/main_data` (the organizer
mount), then a sibling `data/main`:

| Candidate | Setting | What it is |
| --- | --- | --- |
| R1 frozen | `r1="frozen"` | The model fit on round-one data, applied as is. A new machine type gets no type-specific correction. |
| R1+R2 combined | `r1="combined"` | Fit on round-one train + round-two train. Validation and test are round two's. |
| R2 + .25 R1 blend | `r1="blend", r1_weight=.25` | The round-two fit pooled with the frozen round-one model. The weight is set in advance, not tuned. |

The same selection rule applies. A candidate that fails on the new data (for example, an unseen type) is reported and
skipped. `chosen.json` always writes `r1_data: "auto"`, so the ZIP finds `/main_data` on the runner. Blend and combined
fit more data, so measure the cold 2-core runtime before shipping either.

## Leakage audit (2026-11-08)

| What | Where it comes from | Check |
| --- | --- | --- |
| Query features | `load_split` builds each machine's sequence with length = cutoff, so ticks >= cutoff cannot exist (a later row would raise an error, not leak). | By construction |
| Training rows at tick t | Features from `seq[:t]` of every machine in the same episode. Label = the history payout at t. Short-log copies mask `v[:t+1]` and use only ticks < t as inputs. | Test 1: scrambling every machine's ticks >= t leaves features bit-identical |
| Polarity bank | Counts over u < t. The current bit uses ticks t - l < t. Only machines of the same episode. | Test 1b |
| Size menu | Size model sees `s[:t]`, label is the history payout at t < cutoff. | Test 4 |
| Answer files | fit/predict never open validation/test `targets`, `labels`, `conditions`, `reference`. The only label file in the fit path is TRAIN `labels.csv`, read only by adi's `neighbor_adjust` (off). | Test 3 (`--io`): open() spy on a cold fit + predict |
| Missing ticks | -1 is unknown, never a 0 payout. | Test 2 |
| Cross-episode | Features and the polarity bank are built per episode. Machines of other episodes are never inputs. | By construction |
| Validation / test histories | Their PRE-cutoff ticks are used as unlabeled-in-the-contest training rows (history payout at t as label, inputs < t). Validation and test get identical treatment, so validation scores represent test. No query-tick or target information enters. | Allowed by AGENTS.md (histories of other splits) and the kit's rules |
| Selection | `round2.py` reads validation targets outside fit/predict. It chooses on the tuning half; the audit half is only a veto. | Code review |
