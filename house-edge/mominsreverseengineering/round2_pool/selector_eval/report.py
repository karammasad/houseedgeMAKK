"""Read held-out outcomes only after both training decisions exist."""
import argparse,csv,json,pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--own',required=True);p.add_argument('--cross',required=True);p.add_argument('--out',required=True);a=p.parse_args(); root=Path(a.out)
if not (root/'own/decision.json').exists() or not (root/'cross/decision.json').exists():raise SystemExit('both decisions must be frozen before evaluation')
for name,ds in [('own',a.own),('cross',a.cross)]:
 path=Path(ds)/'validation'; d=root/name
 def rows(n):
  with (path/n).open(newline='') as f:return list(csv.DictReader(f))
 def load(n):
  with (d/n/'validation_pred.pkl').open('rb') as f:return pickle.load(f)
 t=load('thanos');b=load('polarity');cond={x['query_id']:x['condition'] for x in rows('conditions.csv')}; by=defaultdict(lambda:defaultdict(list)); cols=('p_0','p_1','p_2','p_5','p_20')
 for r in rows('targets.csv'):
  k=r['query_id'];y=np.array([float(r[c]) for c in cols]);by[cond[k]][r['episode_id']].append([np.sum((np.array(t[k])-y)**2),np.sum((np.array(b[k])-y)**2)])
 groups=[np.array([np.mean(v,axis=0) for v in e.values()]) for e in by.values()]; avg=np.mean([x.mean(axis=0) for x in groups],axis=0);rng=np.random.default_rng(812);z=[]
 for _ in range(5000):
  v=np.mean([x[rng.integers(0,len(x),len(x))].mean(axis=0) for x in groups],axis=0);z.append(v[1]-v[0])
 decision=json.loads((d/'decision.json').read_text());print(name, json.dumps({'chosen':decision['decision'],'thanos_truth_mse':float(avg[0]),'polarity_truth_mse':float(avg[1]),'polarity_minus_thanos':float(avg[1]-avg[0]),'ci95':np.percentile(z,[2.5,97.5]).tolist(),'heldout_episodes':sum(len(x) for x in groups)}))
