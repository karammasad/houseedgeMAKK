# adi main-round upload files: stdlib port of method4_ml v5 (best validation score)

Upload both files to the Main form: `submission.csv` (test forecast) and `entry.zip` (code that recreates it).

- Model: `--method adi.stack` = method4_ml v5 (Kareem: adi gate with size split + signal-shifted priors, method 2
  GateSizes, gradient-boosted stacker). The trees were trained offline with scikit-learn and exported to
  `adi/stack/trees_main.json`; at prediction time they're evaluated in pure Python. **Standard library only**:
  no requirements.txt, no scikit-learn on the runner.
- Validation: tuning 58.241, audit 58.142, full 58.192 / 60. keepcheck vs method2_sizes FINAL (58.074 full,
  upload 56c5cd55): KEEP, P(better) = 0.990, 4/4 conditions better.
- Parity: pure-Python trees = scikit-learn to 3.3e-16 on every validation/test query.
- CSV SHA-256: 4454dc7af5007fc7f318571e2018565a753cb683a8457572f6602ece7f9944f8
- Built and rerun-verified (fresh unzip → byte-identical CSV) with `python -m adi.make_entry --method adi.stack`.
