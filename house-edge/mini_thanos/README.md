# mini_thanos: routed thanos (conservative alternative; the final pick is full thanos, see ../thanos/LOG.md section 11)

Long logs -> `thanos` (exact generator posterior, `../thanos/`); short logs -> the champion (Momin's shared-base) from its saved checkpoint.
The route uses only the observable log shape (fewer than half the pre-cutoff ticks observed = short); no condition labels.

**Why routed** (`../thanos/LOG.md` sections 7-9): thanos beats the champion on long logs on validation AND on independent train outcomes
(P .997, pre-registered rule). Its short-log gain exists only on validation and failed its pre-registered train CV check (P .12), so short logs
stay with the champion.

| | full validation | tuning | audit |
| --- | --- | --- | --- |
| champion (Momin shared-base) | 58.355 | 58.338 | 58.380 |
| **mini_thanos** | **58.467** | 58.414 | 58.527 |
| full thanos (less verified) | 58.629 | 58.494 | 58.785 |

## Run (from `house-edge/`)
    python3 -m mini_thanos.run --data data/main --out build/mini_thanos --ckpt build/main_sb_ckpt/entry_src --also validation

Needs the champion checkpoint (gitignored build artifact): `python3 -m method4_ml.make_entry_ckpt --root mominsreverseengineering --data data/main --out build/main_sb_ckpt`.
Measured end to end (thanos fit on all histories + champion checkpoint predict + thanos predict, test + validation, 2 threads, macOS with another
job running): 564 s wall; peak RSS 215 MB (+138 MB champion process). Test CSV passes `kit/check_submission.py` (sha256 a9c716a6...);
validation reproduces 58.4666. Dependencies: numpy + scipy (thanos) and the champion's pinned scikit-learn stack (checkpoint).
Not done yet: a packaged ZIP with a fresh-unzip rerun check, and a true 2-core container timing.
