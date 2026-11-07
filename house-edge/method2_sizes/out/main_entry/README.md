# Method 2 main-round upload files

Upload both files to the Main form: `submission.csv` (test forecast) and `entry.zip` (code that recreates it).

- Model: adi C5 win gate × method 2 sizes (coupled, tied, learned tie weight, **joint EM**) = `FINAL` in
  `method2_sizes/round2.py`
- Validation: tuning 57.901, audit 57.665, full 57.787 / 60
  (stable .032, short .064, drift .030, shared .027)
- CSV SHA-256: 450755db82c5bbebb5922cb0dd99b2c7d48b0f394ae5fb1c89f61871ab1204f5
- Built and rerun-verified (fresh unzip → byte-identical CSV) with `python -m method2_sizes.make_entry`
- Previous version (no joint EM, full 57.622, SHA d39f5332...) is in git history.
