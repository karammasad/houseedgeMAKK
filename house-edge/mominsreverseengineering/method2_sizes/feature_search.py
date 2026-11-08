"""B1c: which lag / XOR-pair feature best explains a win's size, per machine (train histories only).
Run from repo root: python3 method2_sizes/feature_search.py"""
import sys, itertools, random, math
from collections import defaultdict, Counter
sys.path.insert(0,'.')
from common.data import load_split
from method2_sizes.analysis import g_test, SI
q,seq=load_split('data/main','train')
rng=random.Random(1)
def feat_fns():
    F={}
    for k in range(1,9): F[f'win{k}']=lambda s,t,k=k:(int(s[t-k]>0) if s[t-k]>=0 else None)
    for a,b in itertools.combinations(range(1,9),2):
        F[f'xnor{a}{b}']=lambda s,t,a=a,b=b:(int((s[t-a]>0)==(s[t-b]>0)) if s[t-a]>=0 and s[t-b]>=0 else None)
    return F
F=feat_fns()
tot={}; best=Counter(); per={}
for name,f in F.items():
    G=0
    for r in q:
        s=seq[r['machine_id']]; tab=defaultdict(lambda:[0]*4)
        for t in range(8,len(s)):
            if s[t]>0:
                x=f(s,t)
                if x is not None: tab[x][SI[s[t]]]+=1
        g=g_test(tab)[0]; G+=g; per.setdefault(r['machine_id'],{})[name]=g
    tot[name]=G
for m,d in per.items(): best[max(d,key=d.get)]+=1
print('top features by summed per-machine G (null ~1550-1600):')
for k,v in sorted(tot.items(),key=lambda x:-x[1])[:12]: print(f'  {k}: {v:.0f}')
print('most common best feature per machine:',best.most_common(10))
