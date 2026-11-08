"""Read Thanos directly from a git revision, audit its I/O and measure a fresh run.

No team code is copied to this folder. Run from the repository's working tree.
"""
import argparse
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT))

class GitCode(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, rev): self.rev = rev
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'thanos_method':
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        if fullname.startswith('thanos_method.') and fullname.count('.') == 1:
            return importlib.util.spec_from_loader(fullname, self, is_package=False)
        return None
    def create_module(self, spec): return None
    def exec_module(self, module):
        name = module.__name__
        path = 'house-edge/' + name.replace('.', '/') + ('/__init__.py' if name == 'thanos_method' else '.py')
        source = subprocess.check_output(['git', 'show', f'{self.rev}:{path}'], cwd=REPO, text=True)
        module.__file__ = f'<git:{self.rev}:{path}>'
        exec(compile(source, module.__file__, 'exec'), module.__dict__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--rev', default='360ebc6')
    ap.add_argument('--out', required=True)
    ap.add_argument('--include-train', action='store_true')
    ap.add_argument('--defer-score', action='store_true')
    args = ap.parse_args()
    sys.meta_path.insert(0, GitCode(args.rev))
    from thanos_method.model import Model
    from common.data import public_dir, load_split, write_predictions
    import numpy as np
    import resource
    pub = Path(public_dir(args.data)).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    opened = []
    def hook(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if p.endswith('.csv'): opened.append(str(Path(p).resolve()))
    sys.addaudithook(hook)
    start = time.monotonic(); cpu = time.process_time()
    model = Model(splits='all', tickset='own').fit(str(pub))
    fit_seconds = time.monotonic()-start
    preds = {}
    for sp in (('train', 'validation', 'test') if args.include_train else ('validation', 'test')):
        preds[sp] = model.predict(str(pub), sp)
        q, _ = load_split(str(pub), sp)
        if set(preds[sp]) != {r['query_id'] for r in q}: raise ValueError('query IDs do not match')
        arr = np.asarray(list(preds[sp].values()))
        if arr.shape != (len(q),5) or not np.isfinite(arr).all() or (arr < 0).any() or np.max(np.abs(arr.sum(1)-1)) > 1e-9:
            raise ValueError('invalid probability vector')
    allowed = {str((pub / sp / name).resolve()) for sp in ('train','validation','test') for name in ('queries.csv','history.csv')}
    bad = sorted(set(opened)-allowed)
    if bad: raise ValueError('Unexpected CSV opened during fit/predict: '+repr(bad))
    fit_predict_seconds = time.monotonic()-start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    info = {'revision':args.rev, 'model':'all, own', 'fit_seconds':fit_seconds, 'fit_validation_test_seconds':fit_predict_seconds,
            'cpu_seconds':time.process_time()-cpu, 'peak_rss_bytes':rss if sys.platform == 'darwin' else rss*1024,
            'opened_csv':sorted(set(opened)), 'validation_queries':len(preds['validation']), 'test_queries':len(preds['test'])}
    import pickle
    with open(out/'validation_pred.pkl','wb') as f: pickle.dump(preds['validation'], f)
    if args.include_train:
        with open(out/'train_pred.pkl','wb') as f: pickle.dump(preds['train'], f)
    write_predictions(str(out/'submission.csv'), preds['test'])
    if not args.defer_score:
        from common.evaluate import folds, points
        meta=json.loads((pub/'metadata.json').read_text()); tune, audit=folds(str(pub)); maximum=float(meta['max_points'])
        info['validation_points']={name: points(str(pub), preds['validation'], subset, maximum)[0] for name,subset in [('tuning',tune),('audit',audit),('full',None)]}
    with open(out/'audit.json','w') as f: json.dump(info,f,indent=2)
    print('AUDIT RESULT',json.dumps(info,indent=2),flush=True)

if __name__ == '__main__': main()
