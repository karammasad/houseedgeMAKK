"""Select frozen Thanos versus polarity using training-cutoff outcomes only."""
import argparse,csv,json,pickle
from collections import defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--thanos',required=True);p.add_argument('--polarity',required=True);p.add_argument('--out',required=True);a=p.parse_args()
D=Path(a.data)/'train'; out=Path(a.out)
if out.exists():raise SystemExit('decision exists; refusing a second selection')
def rows(name):
 with (D/name).open(newline='') as f:return list(csv.DictReader(f))
def load(path):
 with open(path,'rb') as f:return pickle.load(f)
t=load(a.thanos); b=load(a.polarity); q=rows('queries.csv'); labels={r['query_id']:int(r['outcome']) for r in rows('labels.csv')}; cond={r['query_id']:r['condition'] for r in rows('conditions.csv')}
if set(t)!=set(b) or set(t)!=set(labels):raise ValueError('query ids differ')
by=defaultdict(lambda:defaultdict(list)); pay=(0,1,2,5,20)
for r in q:
 k=r['query_id']; y=np.array([int(labels[k]==x) for x in pay]); by[cond[k]][r['episode_id']].append(float(np.sum((np.asarray(b[k])-y)**2)-np.sum((np.asarray(t[k])-y)**2)))
groups=[np.array([np.mean(v) for v in d.values()]) for d in by.values()];rng=np.random.default_rng(811); z=np.empty(5000)
for j in range(len(z)):z[j]=np.mean([rng.choice(x,len(x),replace=True).mean() for x in groups])
mean=float(np.mean([x.mean() for x in groups]));ci=np.percentile(z,[2.5,97.5]); decision='polarity' if ci[1]<-.01 else 'thanos'
result={'decision':decision,'baseline':'thanos (only after package verification; otherwise shared-base)', 'train_only':True,'episodes_by_condition':{k:len(v) for k,v in by.items()},'polarity_minus_thanos_brier':mean,'ci95':ci.tolist(),'minimum_brier_improvement':.01,'seed':811,'bootstrap_draws':5000}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
