# Method 2 → team summary (for Person A, Adi / Person C, Person D)

## Recommendation for the main entry
**Adi's C5 win gate × Method 2's sizes (coupled + tied, learned tie weight, joint EM).** Full validation 57.787.

| model | tuning | audit | full val | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| method1_gate (current entry) | 55.054 | 54.795 | 54.927 | .074 | .132 | .086 | .077 |
| adi C5 gate + NPMLE sizes | 55.676 | 55.196 | | | | | |
| method 1 gate + method 2 sizes (fc6b2ab) | 56.999 | 56.974 | 56.985 | .034 | .107 | .039 | .032 |
| adi gate + method 2 sizes (= adi b072457) | 57.685 | 57.427 | 57.560 | .033 | .073 | .033 | .030 |
| + learned tie weight | 57.746 | 57.490 | 57.622 | .033 | .071 | .032 | .029 |
| **+ joint EM (current final)** | **57.901** | **57.665** | **57.787** | **.032** | **.064** | **.030** | **.027** |

Full-validation ratios use the kit metric. Last row vs the row above: paired episode bootstrap P(better) = 1.00,
4/4 conditions better. Entry path (fit + test predict) 138 s, byte-identical rerun, CSV passes the checker.

Two identical ways to run it:
- `--method method2_sizes --kwargs` = `FINAL` in `method2_sizes/round2.py`
- `--method adi` with Adi's sizes="gate" settings plus `"gate_tie_weight": "learn"` (if adi passes it through)

## Packaging: use method2_sizes/make_entry.py (shared tools/make_entry.py left untouched)
`tools/make_entry.py` stages only `common/` and `method*/`, so entries using Adi's gate fail on the runner with
`No module named 'adi'`. `method2_sizes/make_entry.py` is an identical copy that also packages `adi/`
(one-line difference, `EXTRA_PKGS`). Verified end to end: build, kit check, zip, fresh unzip + rerun is
byte-identical (CSV SHA-256 450755db..., about 4.8 min for the whole build):
`python -m method2_sizes.make_entry --method method2_sizes --kwargs "$(python3 -c 'import json,method2_sizes.round2 as r; print(json.dumps(r.FINAL))')" --data data/main --out build/main_m2`
Upload `build/main_m2/submission.csv` + `build/main_m2/entry.zip`. Adi's `--method adi` entries can use this
builder too. Person A: fold the same one-line change into tools/ whenever convenient.

## Dependencies of the recommended model (all standard library)
`method2_sizes/` (sizes.py, model.py), `adi/` (model.py, gate.py), `method1_gate/` (gate.py, model.py:
GateSizes' coupling reads method 1's win-structure posterior; method2's Model subclasses method 1's),
`common/data.py`. Nothing in method3_discovery/ or method4_ml/.

## What I used from each of you
- **Method 1 (Person A):** `gate.py` structures / machine_tables / learn_structure_prior / logsumexp, imported
  read-only for the size coupling (the win-structure posterior with flips marginalized).
- **adi (Adi):** the C5 win gate as the P(win) source (`win_model="adi"`), imported read-only. C0 finding
  (structure ID must be flip-aware) is why the coupling marginalizes flips.
- **Method 4 (Person D):** read only; not used (48.6 tuning, needs scikit-learn).

## What helped / what didn't (method 2 side, keep/revert rule applied)
| change | result |
| --- | --- |
| size gate (fixed per-type big split + binary size state) | kept, +0.98 tuning / +1.58 audit |
| coupling size lag to the win gate (EM table) | kept, +0.63 tuning |
| tied two-state hypotheses | kept, +0.33 tuning; short .113 → .100 |
| stacking adi's gate under these sizes | kept, full +0.58, P=1.00, 4/4 |
| learned tie weight (vs fixed 0.7) | kept, full +0.06, P=1.00, 4/4 |
| blending method 1 gate with adi gate (any weight) | no gain: errors correlate 0.84, best weight = all adi |
| bidirectional coupling (sizes → win structure) | not built: structure uncertainty is only 31% of short win error |

## Where the remaining error is (`python -m method2_sizes.decompose`)
79% of the remaining error is on the WIN side (WIN .042 vs SIZE .011 of R .053). 14 of the 20 worst queries are
**jackpot machines with the structure identified at posterior 1.00 but the win RATE wrong** (true win rates are
continuous per machine: jackpot "on" spans 0.27–0.85). Next lever, for Person A / Adi: better per-machine win-rate
estimates (hierarchical on-rate prior per type × signal, or recency on the on-cell rate), not structure search.
Suggested for Adi: feed adi's win-structure posterior into GateSizes' coupling (hook: `GateSizes._win_posterior`).

## Round two (9 PM)
`python -m method2_sizes.round2 --data data/adaptation --baseline data/main --out build/round2`
refits, reruns discovery and prints a learned-structure diff with trust flags. Tested: the free-cell fallback turns
on automatically (tie weight 0.97 on real data → 0.02 on synthetic continuous sizes, 0.29 on a synthetic
three-level gate, whose lag 3 it rediscovered).
