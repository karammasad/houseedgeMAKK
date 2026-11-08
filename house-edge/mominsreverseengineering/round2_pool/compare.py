"""Compare two fixed prediction sets using tuning, then one audit veto."""
import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.data import public_dir
from common.evaluate import CONDITIONS, folds, points


def decide_tuning(base, candidate, margin, worst_limit):
    base_score, base_ratios = base
    candidate_score, candidate_ratios = candidate
    bad = max(candidate_ratios[c] / max(base_ratios[c], 1e-12) - 1 for c in CONDITIONS)
    return candidate_score - base_score >= margin and bad <= worst_limit, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--base', required=True, help='baseline validation_pred.pkl')
    ap.add_argument('--candidate', required=True, help='candidate validation_pred.pkl')
    ap.add_argument('--out', required=True, help='new JSON decision file')
    ap.add_argument('--allow-main', action='store_true', help='rehearsal only; audit on main is already consumed')
    args = ap.parse_args()
    pub = Path(public_dir(args.data)).resolve()
    meta = json.loads((pub / 'metadata.json').read_text())
    if meta.get('round') != 'adaptation' and not args.allow_main:
        ap.error('expected adaptation data; --allow-main permits a rehearsal')
    out = Path(args.out)
    if out.exists():
        ap.error('decision file already exists; keep the original audit decision')
    with open(args.base, 'rb') as f: base = pickle.load(f)
    with open(args.candidate, 'rb') as f: candidate = pickle.load(f)
    if set(base) != set(candidate):
        raise ValueError('prediction IDs differ')
    with (pub / 'validation' / 'queries.csv').open(newline='') as f:
        ids = [r['query_id'] for r in csv.DictReader(f)]
    if len(ids) != len(set(ids)) or set(ids) != set(base):
        raise ValueError('validation query IDs do not match predictions')
    for predictions in (base, candidate):
        for query_id, vector in predictions.items():
            if len(vector) != 5 or any(not math.isfinite(x) or x < 0 for x in vector) or abs(sum(vector)-1) > 1e-6:
                raise ValueError(f'invalid probabilities for {query_id}')
    maxp = float(meta['max_points'])
    tune, audit = folds(str(pub))
    b = points(str(pub), base, tune, maxp)
    c = points(str(pub), candidate, tune, maxp)
    margin = 0.15 * maxp / 60.0
    passes, worst = decide_tuning(b, c, margin, 0.10)
    result = {'chosen': 'base', 'tuning': {'base': b[0], 'candidate': c[0],
              'gain': c[0]-b[0], 'margin': margin, 'worst_condition_fraction': worst}, 'audit': None}
    print('Tuning:', json.dumps(result['tuning']), flush=True)
    if passes:
        # Candidate is frozen before this sole audit comparison.
        ab = points(str(pub), base, audit, maxp)
        ac = points(str(pub), candidate, audit, maxp)
        condition_ok = all(ac[1][k] <= ab[1][k] for k in CONDITIONS)
        result['audit'] = {'base': ab[0], 'candidate': ac[0], 'gain': ac[0]-ab[0],
                           'base_ratios': ab[1], 'candidate_ratios': ac[1], 'all_conditions_not_worse': condition_ok}
        if ac[0] - ab[0] >= margin and condition_ok:
            result['chosen'] = 'candidate'
        print('Audit (one frozen comparison):', json.dumps(result['audit']), flush=True)
    else:
        print('Candidate did not clear tuning gate; audit untouched.', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f: json.dump(result, f, indent=2)
    print('Chosen:', result['chosen'], '->', out)


if __name__ == '__main__':
    main()
