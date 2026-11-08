"""Fresh synthetic stress: only the win-gate input moves to the adjacent machine.

Separate episode seeds, held-out query probabilities; not an organizer simulator.
"""
import csv, json, math, random, sys
from pathlib import Path

OUT=Path(sys.argv[1]); seed=int(sys.argv[2]) if len(sys.argv)>2 else 2026110801; rng=random.Random(seed)
types=('steady','balanced','jackpot'); conds=('stable','short_history','drift','shared_change')
V={ 'steady':([.76,.18,.055,.005],[.45,.42,.12,.01]),'balanced':([.50,.34,.13,.03],[.22,.53,.21,.04]),'jackpot':([.34,.24,.31,.11],[.12,.33,.41,.14]) }
rates={'steady':(.94,.12),'balanced':(.81,.07),'jackpot':(.64,.03)}
def sig(x): return 1/(1+math.exp(-x))
def write(path, cols, rows):
 path.parent.mkdir(parents=True, exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.writer(f); w.writerow(cols); w.writerows(rows)
def gen(split, n):
 Q=[]; H=[]; C=[]; T=[]; L=[]; base=[]
 for e in range(n):
  cond=conds[e%4]; eid=f'{split}-e{e:03d}'; cut=rng.randint(200,245); ms=[]; pay=[[] for _ in range(8)]
  for m in range(8):
   ty=types[(e*8+m)%3]; signal=rng.choice((-1,0,1)); on,off=rates[ty]; offs=[math.log(x/(1-x))+0.3*signal+rng.gauss(0,.3) for x in (off,on)]
   flip=None if cond in ('stable','short_history') else (int(cut*.68) if cond=='shared_change' else rng.randint(int(cut*.55),int(cut*.8)))
   ms.append((ty,signal,offs,rng.randrange(2),rng.randrange(2),flip))
  for tick in range(cut+1):
   pv=[]
   for m,(ty,signal,offs,pol,direction,flip) in enumerate(ms):
    # Both rules and payout sizes are deterministic functions of earlier ticks.
    bit=int(pay[(m+1)%8][tick-3]>0) if tick>=3 else 0
    state=int(pay[m][tick-1]>=2) ^ direction if tick>=1 else direction
    gate=bit ^ pol ^ (flip is not None and tick>=flip)
    pw=sig(offs[gate]+.4*(2*state-1)); v=V[ty][state]
    pv.append([1-pw]+[pw*x for x in v])
   if tick==cut: break
   for m in range(8): pay[m].append(rng.choices((0,1,2,5,20),pv[m])[0])
  for m,(ty,signal,*_) in enumerate(ms):
   mid=f'{eid}-m{m}'; qid=mid+'-q'; Q.append([qid,eid,mid,ty,signal,cut]); L.append([qid,rng.choices((0,1,2,5,20),pv[m])[0]]); C.append([qid,cond]); T.append([qid,eid,*pv[m],L[-1][1]])
   keep=set(range(cut-40,cut)) | set(rng.sample(range(cut-40),24)) if cond=='short_history' else set(range(cut))
   H.extend([eid,mid,t,pay[m][t]] for t in sorted(keep))
   nobs=len(keep); counts=[sum(pay[m][t]==v for t in keep) for v in (0,1,2,5,20)]; base.append([qid]+[(counts[i]+4)/(nobs+20) for i in range(5)])
 d=OUT/split
 write(d/'queries.csv',['query_id','episode_id','machine_id','machine_type','signal','cutoff'],Q)
 write(d/'history.csv',['episode_id','machine_id','tick','payout'],H)
 if split!='test':
  write(d/'labels.csv',['query_id','outcome'],L); write(d/'conditions.csv',['query_id','condition'],C)
 if split=='validation':
  write(d/'targets.csv',['query_id','episode_id','p_0','p_1','p_2','p_5','p_20','outcome'],T); write(d/'reference.csv',['query_id','p_0','p_1','p_2','p_5','p_20'],base)
for sp,n in [('train',32),('validation',32),('test',8)]: gen(sp,n)
(OUT/'metadata.json').write_text(json.dumps({'round':'adaptation','max_points':40,'season_id':f'synthetic-cross-{seed}','seed':seed,'synthetic':True}))
print('Generated fresh held-out synthetic cross-machine kit:',OUT)
