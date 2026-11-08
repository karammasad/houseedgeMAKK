#!/bin/bash
# Copy a freshly-unzipped adaptation kit's public/ folder into data/adaptation/, with sanity checks.
# Usage: round2/setup_kit.sh /path/to/unzipped_kit_or_its_public_dir
set -euo pipefail
cd "$(dirname "$0")/.."   # house-edge/ (this worktree root)

SRC="${1:?Usage: round2/setup_kit.sh /path/to/unzipped_kit (containing public/, or the public/ dir itself)}"

if [ -f "$SRC/metadata.json" ]; then
  PUB="$SRC"
elif [ -f "$SRC/public/metadata.json" ]; then
  PUB="$SRC/public"
else
  echo "ERROR: no metadata.json found at $SRC or $SRC/public" >&2
  exit 1
fi

ROUND=$(python3 -c "import json;print(json.load(open('$PUB/metadata.json')).get('round'))")
echo "kit round = $ROUND"
if [ "$ROUND" != "adaptation" ]; then
  echo "WARNING: metadata round is '$ROUND', not 'adaptation' — double check this is the right kit." >&2
fi

mkdir -p data
if [ -e data/adaptation ]; then
  echo "ERROR: data/adaptation already exists — refusing to overwrite. Remove it first if you mean to replace it." >&2
  exit 1
fi
cp -R "$PUB" data/adaptation
echo "copied $PUB -> data/adaptation/"
for sp in train validation test; do
  n=$(python3 -c "import csv;print(sum(1 for _ in csv.DictReader(open('data/adaptation/$sp/queries.csv'))))" 2>/dev/null || echo "MISSING")
  echo "  $sp/queries.csv: $n rows"
done
python3 -c "import json; print(json.dumps(json.load(open('data/adaptation/metadata.json')), indent=1))"
