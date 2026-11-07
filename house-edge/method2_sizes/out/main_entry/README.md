# Method 2 main-round upload files

Upload both files to the Main form: `submission.csv` (test forecast) and `entry.zip` (code that recreates it).

- Model: adi C5 win gate **+ size split (adi round 4) + signal-slope cell prior (adi round 5)** × method 2 sizes
  (coupled, tied, learned tie weight, joint EM) = `FINAL` in `method2_sizes/round2.py`
  (`ADI` has `"size_split": True, "cell_signal": True`)
- Validation: tuning 58.156, audit 57.986, full 58.074 / 60
  (stable .021, short .062, drift .026, shared .024). keepcheck vs the previous FINAL: KEEP,
  bootstrap P(better) = 0.999, 4/4 conditions better.
- CSV SHA-256: 56c5cd558ce950c2b736787692c907006b6e70d02c718a8595bb69947ed07f2e
- Built and rerun-verified (fresh unzip → byte-identical CSV) with `python -m method2_sizes.make_entry`
- Previous versions are in git history: size split only (full 57.960, SHA e2686e96...), joint EM without size
  split (full 57.787, SHA 450755db...), no joint EM (full 57.622, SHA d39f5332...).
