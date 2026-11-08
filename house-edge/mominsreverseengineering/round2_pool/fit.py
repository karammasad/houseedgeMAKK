"""Fit one explicitly named adaptation candidate without reading validation answers."""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.data import public_dir, write_predictions
from method4_ml.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--kwargs', default='{}')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--allow-rehearsal', action='store_true')
    args = parser.parse_args()
    pub = public_dir(os.path.abspath(args.data))
    with open(os.path.join(pub, 'metadata.json')) as f:
        metadata = json.load(f)
    if metadata.get('round') != 'adaptation' and not args.allow_rehearsal:
        parser.error('expected adaptation data; use --allow-rehearsal for other kits')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    kw = json.loads(args.kwargs)
    if 'cache' in kw or 'workers' in kw:
        parser.error('set workers with --workers; cache is always disabled')
    start = time.time()
    print('Fitting', json.dumps(kw), 'on', pub, flush=True)
    model = Model(cache=False, workers=args.workers, **kw).fit(pub)
    valid = model.predict(pub, 'validation')
    test = model.predict(pub, 'test')
    model.__dict__.pop('_sz', None)
    with (out / 'model.pkl').open('wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (out / 'validation_pred.pkl').open('wb') as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_predictions(str(out / 'submission.csv'), test)
    with (out / 'fit.json').open('w') as f:
        json.dump({'kwargs': kw, 'seconds': round(time.time()-start, 1), 'data': pub}, f, indent=2)
    print('Finished fit + validation + test in', round(time.time()-start), 's:', out, flush=True)


if __name__ == '__main__':
    main()
