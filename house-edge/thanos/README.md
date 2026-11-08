# thanos: exact generator posterior (EGP)

One joint probabilistic model of each machine's payout stream; the forecast is its exact posterior predictive at the cutoff.
No gate code, no tree stacker, no logistic correction, no size snap, no imports from adi / method1-4 / mominsreverseengineering.
Needs only numpy + scipy (pinned in `method4_ml/requirements.txt`). Design and pre-registration: `DESIGN.md`; every run and verdict: `LOG.md`.

## Model in one paragraph
Per machine, latent and summed exactly: gate structure (XOR lag pair) + size lag (support learned by screening, not hardcoded), size direction and
win-shift direction (joint prior learned: they are tied ~90%), polarity, and one flip time (or none; prior learned separately for short / long logs).
Base win logits per gate side are integrated on a grid against per-type normal priors shifted by `signal`; the win rate also moves by +-gamma with
the size state; given a win the size follows one of two 4-vectors per type. Every hyperparameter is fitted by maximum marginal likelihood on
HISTORIES only (labels, targets and conditions are never opened).

## Commands (from `house-edge/`)
    python3 -m thanos.run --data data/main --splits all --out build/thanos --also validation   # entry: fit + test CSV (~6-11 min, 2 threads)
    python3 -m thanos.run --data data/main --splits train --out build/thanos_ind               # strict inductive (train histories only)
    python3 -m thanos.tests_leakage --data data/main                                           # 8 leakage / validity tests
    python3 -m thanos.cv --data data/main --out build/thanos_cv                                # train-only episode-grouped CV
    python3 -m thanos.compare --new <val.pkl> --old <champion_val.pkl>                         # official metric + paired bootstrap
`--tickset common` reproduces the pre-registered first version (section 1 of LOG.md).

## Results (full validation, official metric; see LOG.md for the caveats)
| protocol | champion (Momin shared-base) | thanos | change | P(gain) |
| --- | --- | --- | --- | --- |
| transductive (histories of all splits) | 58.355 | **58.629** | +0.274 | .994 |
| inductive (train histories only) | 58.332 | **58.567** | +0.235 | .982 |

Better in all four conditions (stable -8%, short -21%, drift -12%, shared -20% error) and both kit halves. The per-structure tick set (the change that
fixed short logs) was made AFTER the first validation look, so validation alone does not prove it; the train-only checks in LOG.md decide.
Runtime: 689 s wall / 600 CPU-s for fit + test + validation (transductive, 2 threads, machine shared with 3 other jobs); peak RSS 255 MB.

## Status (see LOG.md sections 7-9)
- Long logs: thanos beats the champion on validation AND on independent train outcomes (P .997). Established.
- Short logs: the tick-set fix wins on validation but train-only CV fails its pre-registered rule (P .12) and train evidence is inconclusive.
- Recommended conservative entry: ROUTED (thanos on long logs, champion checkpoint on short logs): 58.467 validation (+0.11).
  Full thanos (58.629) is the higher-upside, less-verified option.
