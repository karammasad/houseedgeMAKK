#!/bin/bash
# Round-two run: a verified shared-base fallback first, then a polarity candidate.
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash round2_pool/run.sh /absolute/path/to/public /absolute/path/to/new-output [workers]" >&2
  exit 2
fi
DATA="$1"
OUT="$2"
WORKERS="${3:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "$DATA" in /*) ;; *) echo 'Use an absolute data path' >&2; exit 2;; esac
case "$OUT" in /*) ;; *) echo 'Use an absolute output path' >&2; exit 2;; esac
if [ -e "$OUT" ]; then echo "Output already exists: $OUT" >&2; exit 2; fi
python3 round2_pool/preflight.py --data "$DATA"
mkdir -p "$OUT"
printf '\n[1/5] Fit shared-base fallback\n'
python3 round2_pool/fit.py --data "$DATA" --out "$OUT/base" --workers "$WORKERS" --kwargs '{"shared_base":true}'
printf '\n[2/5] Build and verify fallback; upload these files as soon as ready\n'
python3 -m method4_ml.make_entry --data "$DATA" --out "$OUT/fallback_entry" --model-pkl "$OUT/base/model.pkl" --kwargs '{"shared_base":true}'
printf '\nFALLBACK READY: %s/fallback_entry/submission.csv and %s/fallback_entry/entry.zip\n' "$OUT" "$OUT"
printf '\n[3/5] Fit cross-machine polarity candidate\n'
if ! python3 round2_pool/fit.py --data "$DATA" --out "$OUT/polarity" --workers "$WORKERS" --kwargs '{"shared_base":true,"polarity":true,"lr_spec":["d","chg","pol"]}'; then
  echo 'Polarity fit failed; use the verified fallback above.' >&2
  exit 1
fi
printf '\n[4/5] Compare fixed candidates\n'
python3 round2_pool/compare.py --data "$DATA" --base "$OUT/base/validation_pred.pkl" --candidate "$OUT/polarity/validation_pred.pkl" --out "$OUT/decision.json"
CHOICE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' "$OUT/decision.json")"
if [ "$CHOICE" = candidate ]; then
  printf '\n[5/5] Build and verify selected polarity candidate\n'
  python3 -m method4_ml.make_entry --data "$DATA" --out "$OUT/selected_entry" --model-pkl "$OUT/polarity/model.pkl" --kwargs '{"shared_base":true,"polarity":true,"lr_spec":["d","chg","pol"]}'
  printf '\nSELECTED: %s/selected_entry/submission.csv and %s/selected_entry/entry.zip\n' "$OUT" "$OUT"
else
  printf '\nSELECTED FALLBACK: %s/fallback_entry/submission.csv and %s/fallback_entry/entry.zip\n' "$OUT" "$OUT"
fi
