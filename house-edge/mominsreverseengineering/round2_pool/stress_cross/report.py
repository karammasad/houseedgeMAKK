"""Report pre-specified held-out cross-machine stress comparison by episode."""
import csv, pickle, sys
from collections import defaultdict
import numpy as np
from pathlib import Path
root=Path(sys.argv[1]); data=Path(sys.argv[2]); val=data/'validation'
def rows(name):
 with (val/name).open(newline='') as f: return list(csv.DictReader(f))
with (root/'cross_stress_base/validation_pred.pkl').open('rb') as f: base=pickle.load(f)
with (root/'cross_stress_polarity/validation_pred.pkl').open('rb') as f: polar=pickle.load(f)
with (root/'cross_stress_thanos/validation_pred.pkl').open('rb') as f: thanos=pickle.load(f)
q=rows('queries.csv'); h=rows('history.csv'); n=defaultdict(int)
for row in h:n[row['machine_id']]+=1
route={r['query_id']: (base if n[r['machine_id']] < int(r['cutoff'])/2 else thanos)[r['query_id']] for r in q}
condition={r['query_id']:r['condition'] for r in rows('conditions.csv')}; eps=defaultdict(lambda:defaultdict(list)); cols=('p_0','p_1','p_2','p_5','p_20')
for r in rows('targets.csv'):
 k=r['query_id']; y=np.array([float(r[c]) for c in cols]); eps[condition[k]][r['episode_id']].append([sum((np.array(p[k])-y)**2) for p in (route,polar,thanos,base)])
groups={c:np.array([np.mean(v,axis=0) for v in e.values()]) for c,e in eps.items()}
print('Held-out episodes per condition:',{c:len(a) for c,a in groups.items()})
print('Mean five-class squared error: Mini, polarity, Thanos, shared-base')
for c,x in groups.items():print(c, np.round(x.mean(axis=0),6))
a=np.array([x.mean(axis=0) for x in groups.values()]).mean(axis=0)
print('Equal-condition average:',np.round(a,6))
rng=np.random.default_rng(801); samples=[]
for _ in range(5000):
 v=np.array([x[rng.integers(0,len(x),len(x))].mean(axis=0) for x in groups.values()]).mean(axis=0)
 samples.append([v[0]-v[1],v[0]-v[2],v[0]-v[3]])
samples=np.asarray(samples)
for j,name in enumerate(('polarity','Thanos','shared-base')):
 print('Mini minus',name,'error:',round(float(a[0]-a[j+1]),6),'95% episode-bootstrap CI:',np.round(np.percentile(samples[:,j],[2.5,97.5]),6),'P(Mini worse):',round(float(np.mean(samples[:,j]>0)),4))
