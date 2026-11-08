"""Check an adaptation kit and runtime without opening any answer files."""
import argparse
import csv
import importlib.metadata
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--allow-rehearsal', action='store_true')
    args = ap.parse_args()
    pub = Path(args.data).resolve()
    if (pub / 'public').is_dir(): pub /= 'public'
    meta = json.loads((pub / 'metadata.json').read_text())
    if meta.get('round') != 'adaptation' and not args.allow_rehearsal:
        ap.error('expected round=adaptation; use --allow-rehearsal for a non-adaptation kit')
    print('Kit:', {k: meta.get(k) for k in ('round', 'season_id', 'max_points', 'data_version')})
    for package in ('scikit-learn', 'numpy', 'scipy', 'joblib', 'threadpoolctl'):
        try: version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: raise SystemExit(f'Missing dependency: {package}')
        print(f'{package}=={version}')
    for split in ('train', 'validation', 'test'):
        with (pub / split / 'queries.csv').open(newline='') as f: rows = list(csv.DictReader(f))
        cuts = {r['machine_id']: int(r['cutoff']) for r in rows}
        if len(cuts) != len(rows): raise ValueError(f'{split}: duplicate machine ID')
        seen, last, types = Counter(), defaultdict(lambda: -1), Counter(r['machine_type'] for r in rows)
        with (pub / split / 'history.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                m, t = r['machine_id'], int(r['tick'])
                if m not in cuts or t < 0 or t >= cuts[m]: raise ValueError(f'{split}: invalid history tick')
                seen[m] += 1
                last[m] = max(last[m], t)
        short = sum(seen[m] < cuts[m] / 2 for m in cuts)
        print(f'{split}: queries={len(rows)} types={dict(types)} short={short} cutoffs={min(cuts.values())}-{max(cuts.values())}')
    missing = [x for x in ('targets.csv', 'conditions.csv', 'reference.csv') if not (pub / 'validation' / x).is_file()]
    if missing: raise SystemExit('Missing files needed for selection: ' + ', '.join(missing))
    print('Preflight OK. Only metadata, queries, and histories were read; validation answers were not opened.')


if __name__ == '__main__': main()
