import csv, collections, hashlib, math
pub="/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-rEZWCR/e0c65641-1400-4a02-8422-550c7f4089b5/scratchpad/r2kit/public"
rd=lambda p:{r["query_id"]:[float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in csv.DictReader(open(p))}
A=rd(pub+"/../../adi/validation.csv"); T=rd(pub+"/validation/targets.csv"); R=rd(pub+"/validation/reference.csv")
cond={r["query_id"]:r["condition"] for r in csv.DictReader(open(pub+"/validation/conditions.csv"))}
Q={r["query_id"]:r for r in csv.DictReader(open(pub+"/validation/queries.csv"))}
C=["stable","short_history","drift","shared_change"]
byep=collections.defaultdict(list)
for q in T: byep[Q[q]["episode_id"]].append(q)
ec={e:cond[v[0]] for e,v in byep.items()}
tune,audit=set(),set()
for c in C:
    o=sorted([e for e in byep if ec[e]==c],key=lambda e:hashlib.sha256(e.encode()).hexdigest()); tune|=set(o[::2]); audit|=set(o[1::2])
se=lambda p,t:sum((a-b)**2 for a,b in zip(p,t))
tot=sum(se(A[q],T[q]) for q in T)
# 1. calibration of P(win) by predicted bin
print("pred P(win) bin | n | mean pred | mean true | share of err")
for lo in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]:
    s=[q for q in T if lo<=1-A[q][0]<lo+.1]
    if s: print(f"  {lo:.1f}-{lo+.1:.1f} n={len(s):4d} pred {sum(1-A[q][0] for q in s)/len(s):.3f} true {sum(1-T[q][0] for q in s)/len(s):.3f}  err share {sum(se(A[q],T[q]) for q in s)/tot:.2f}  pwin-err share {sum((A[q][0]-T[q][0])**2 for q in s)/tot:.2f}")
# share of error in p0 vs size mix
p0=sum((A[q][0]-T[q][0])**2 for q in T); print(f"share of total err from p0 alone {p0/tot:.2f}")
# conditional on intermediate: is true systematically further from .5 (underconfident) or closer (overconfident)?
mid=[q for q in T if .3<=1-A[q][0]<.7]
d=[( (1-T[q][0])-(1-A[q][0]) )*(1 if 1-A[q][0]>=.5 else -1) for q in mid]
print(f"mid n={len(mid)}: mean signed (true-pred) toward extreme {sum(d)/len(d):+.4f}  frac true more extreme {sum(x>0 for x in d)/len(d):.2f}")
def adj(p,s,center):
    w=1-p[0]; w=min(max(w,1e-6),1-1e-6); z=math.log(w/(1-w)); zc=math.log(center/(1-center))
    w2=1/(1+math.exp(-(zc+s*(z-zc)))); m=sum(p[1:]); return [1-w2]+[x/m*w2 for x in p[1:]] if m>0 else p
def pts(M,eps):
    r=0
    for c in C:
        es=[e for e in eps if ec[e]==c]
        E=sum(sum(se(M[q],T[q]) for q in byep[e])/len(byep[e]) for e in es)/len(es); B=sum(sum(se(R[q],T[q]) for q in byep[e])/len(byep[e]) for e in es)/len(es); r+=E/B/4
    return 20/(1+r)
base_t=pts(A,tune); print(f"tuning base {base_t:.4f}")
for s in (0.8,0.9,0.95,1.05,1.1,1.2,1.35):
    M={q:adj(A[q],s,.5) for q in T}; print(f"  temp s={s}: tuning {pts(M,tune)-base_t:+.4f}")
