"""Freeze a training-only diagnostic decision before accessing held-out outcomes."""
import csv,json,pickle,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
base=Path(sys.argv[1]); data=Path(sys.argv[2]); output=Path(sys.argv[3]); train=data/'train'
if output.exists():raise SystemExit('decision already exists')
def rd(n):
 with (train/n).open(newline='') as f:return list(csv.DictReader(f))
def pred(n):
 with (base/n/'train_pred.pkl').open('rb') as f:return pickle.load(f)
a,b,t=pred('cross_stress_base'),pred('cross_stress_polarity'),pred('cross_stress_thanos')
labels={r['query_id']:int(r['outcome']) for r in rd('labels.csv')}; q=rd('queries.csv'); observed=defaultdict(int)
for r in rd('history.csv'):observed[r['machine_id']]+=1
by=defaultdict(list); payouts=(0,1,2,5,20)
for r in q:
 k=r['query_id']; y=np.array([int(labels[k]==x) for x in payouts]); mini=np.array(a[k] if observed[r['machine_id']] < int(r['cutoff'])/2 else t[k]); polar=np.array(b[k]); by[r['episode_id']].append(float(sum((polar-y)**2)-sum((mini-y)**2)))
d=np.array([np.mean(v) for v in by.values()]); rng=np.random.default_rng(801); z=rng.choice(d,size=(5000,len(d)),replace=True).mean(axis=1); ci=np.percentile(z,[2.5,97.5]); choice='polarity' if ci[1]<0 else 'mini'
res={'train_only':True,'episodes':len(d),'polarity_minus_mini_outcome_brier':float(d.mean()),'ci95':ci.tolist(),'choice':choice,'rule':'choose polarity only if episode bootstrap 95% CI excludes zero in its favor'}
output.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
