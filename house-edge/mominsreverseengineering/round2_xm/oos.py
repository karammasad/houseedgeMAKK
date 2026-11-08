"""Label-free check on HISTORIES: at tick t (long logs, t>=48), regress own win w_t on own features
[own rate over ticks<t, own win rate over t-8..t-1] plus peer feature f_t (same definition as xm.py, with 'cutoff' = t).
Placebo: peer feature taken from a different episode (same split, same t). Coefficient on f and its t-stat."""
import csv, sys, collections, numpy as np
P=sys.argv[1]
for sp in ("train","validation","test"):
    qs=list(csv.DictReader(open(f"{P}/{sp}/queries.csv"))); H=collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{P}/{sp}/history.csv")): H[(r["episode_id"],r["machine_id"])][int(r["tick"])]=int(r["payout"])>0
    ep=collections.defaultdict(list)
    for q in qs: ep[q["episode_id"]].append((q["episode_id"],q["machine_id"]))
    eps=sorted(ep); W={}
    for k,h in H.items():
        c=max(h)+1; a=np.full(c,-1); 
        for t,v in h.items(): a[t]=v
        W[k]=a
    def mats(e_peer_of):
        X=[];y=[]
        for e in eps:
            ms=ep[e]; L=min(len(W[m]) for m in ms+ep[e_peer_of[e]])
            if any((W[m]<0).any() for m in ms): continue      # long logs only (fully observed)
            cs={m:np.concatenate([[0],np.cumsum(W[m]==1)]) for m in ep[e_peer_of[e]]+ms}
            for t in range(48,L):
                for m in ms:
                    own_base=cs[m][t]/t; own_rec=(cs[m][t]-cs[m][t-8])/8
                    peers=[p for p in ep[e_peer_of[e]] if p!=m]
                    f=np.mean([(cs[p][t]-cs[p][t-8])/8-cs[p][t]/t for p in peers])
                    X.append([1,own_base,own_rec,f]); y.append(W[m][t])
        X=np.array(X,float); y=np.array(y,float); b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
        cov=np.linalg.inv(X.T@X)*(r@r)/(len(y)-4); return b[3], b[3]/np.sqrt(cov[3,3]), len(y)
    real={e:e for e in eps}; plac={e:eps[(i+1)%len(eps)] for i,e in enumerate(eps)}
    b,tt,n=mats(real); bp,tp,_=mats(plac)
    print(f"{sp:10s} n={n:6d}  peer coef {b:+.3f} (t={tt:+.1f})   placebo (other episode) {bp:+.3f} (t={tp:+.1f})")
