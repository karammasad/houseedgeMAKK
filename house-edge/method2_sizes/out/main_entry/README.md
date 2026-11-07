# Method 2 main-round upload files

Upload both files to the Main form: `submission.csv` (test forecast) and `entry.zip` (code that recreates it).

- Model: adi C5 win gate **+ size split (adi round 4)** × method 2 sizes (coupled, tied, learned tie weight,
  joint EM) = `FINAL` in `method2_sizes/round2.py` (`ADI` now has `"size_split": True`)
- Validation: tuning 58.080, audit 57.832, full 57.960 / 60
  (stable .024, short .064, drift .027, shared .025). keepcheck vs the previous FINAL: KEEP,
  bootstrap P(better) = 1.000, 4/4 conditions better.
- CSV SHA-256: e2686e96d70b39b242d6d7afc913366ced004670406b187c35083b2635f8b789
- Built and rerun-verified (fresh unzip → byte-identical CSV) with `python -m method2_sizes.make_entry`
- Previous versions are in git history: joint EM without size split (full 57.787, SHA 450755db...),
  no joint EM (full 57.622, SHA d39f5332...).
