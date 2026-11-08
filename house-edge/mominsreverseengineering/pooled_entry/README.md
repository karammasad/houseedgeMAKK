# Shared-base + cook main-round entry

This is the verified 50/50 pool of Momin's shared-base model and Adi's cook. `entry.zip` contains the two fitted checkpoints and their matching code, isolated in separate Python processes. `submission.csv` is the matching main-round test forecast.

Main validation from the saved member forecasts: tuning 58.3614, audit 58.3821, full 58.3689 / 60. The audit half was previously consumed; the incremental gain over shared-base alone is inconclusive. These figures are not a test-set score.

The kit checker validated all 1,536 test rows. A fresh unzip reproduced the CSV byte for byte; the initial packaged prediction took 242 seconds. CSV SHA-256: `0d0984e5a344fd8f723e80415aad65be0fdbb3df669c434efcfe4eec26025642`.

To reproduce with the same installed package versions listed inside the ZIP:

```sh
unzip entry.zip -d /tmp/houseedge_pool_entry
python /tmp/houseedge_pool_entry/predict.py --data /path/to/main/public --out /tmp/houseedge_pool.csv
```

This checkpoint is fitted to the main-round data. It does not refit for adaptation. Adi's reusable two-process builder is `house-edge/adi/pool2_entry.py` on main.
