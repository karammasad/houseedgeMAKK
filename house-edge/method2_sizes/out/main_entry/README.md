# Method 2 main-round upload files

Upload both files to the Main form: `submission.csv` (test forecast) and `entry.zip` (code that recreates it).

- Model: adi C5 win gate × method 2 sizes (coupled, tied, learned tie weight) = `FINAL` in `method2_sizes/round2.py`
- Validation: tuning 57.746, audit 57.490, full 57.622 / 60
- CSV SHA-256: d39f5332295f3ed101cd815cc16b611c481e98746db5385436f6d08a5b583775
- Built and rerun-verified (fresh unzip → byte-identical CSV) with `python -m method2_sizes.make_entry`
