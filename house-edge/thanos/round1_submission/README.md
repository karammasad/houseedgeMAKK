# Round 1 submission: full thanos (upload BOTH files to the Round 1 Google Form)
- `submission.csv`  sha256 48e008791ac00b437c62c0f7c6de7f2c3a3eaee482268b4327388b34c61f7b04  (kit checker: VALID FORMAT, 1536 queries, main, test)
- `entry.zip`       sha256 2a4e0a63b88b785b... (Momin's deploy_thanos package, pins numpy 2.1.3 / scipy 1.18.1 to match the CSV's environment)
Verified: entry.zip unzipped into a fresh folder and rerun (python predict.py --data data/main --out ...) reproduces submission.csv BYTE-IDENTICALLY
(464 s wall, 202 MB). Full validation of this model: 58.629 (champion 58.355). Decision: ../LOG.md section 11.
