# adi main-round upload files: adi FINAL + size-state win correction (best validation score)

Upload both files to the Main form: `submission.csv` (test forecast) and `entry.zip` (code that recreates it).

- Model: `--method adi.sizewin` = the team FINAL (adi gate with size split + signal prior, method 2 sizes) plus a
  6-number logistic correction that moves P(win) with the size model's estimated hidden size state (adi round 7,
  inspired by method4_ml v5). Standard library only; fully refits from the data (no offline step).
- Validation: tuning 58.221, audit 58.211, full 58.214 / 60. keepcheck vs FINAL (58.074 full, upload 56c5cd55):
  KEEP, P(better) = 0.992, 4/4 conditions better. (The stdlib port of method4 v5, adi.stack, scores 58.192 full.)
- CSV SHA-256: 1ddc8832b7a519d2b1a59fc52a8b9d4db91570b76ae31ed3a3365f1f03269b84
- Built and rerun-verified (fresh unzip → byte-identical CSV) with `python -m adi.make_entry --method adi.sizewin`.
- Previous file in this folder: adi.stack (method4 v5 port, 58.192 full, SHA 4454dc7a...), in git history.
