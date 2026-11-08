import csv, collections, hashlib
pub="/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-rEZWCR/e0c65641-1400-4a02-8422-550c7f4089b5/scratchpad/r2kit/public"
rd=lambda p:{r["query_id"]:[float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in csv.DictReader(open(p))}
A=rd(pub+"/../../adi/validation.csv"); Cm=rd(pub+"/../../r2work/out_common/validation.csv"); T=rd(pub+"/validation/targets.csv"); R=rd(pub+"/validation/reference.csv")
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
def pts(M,eps):
    r=0;rat={}
    for c in C:
        es=[e for e in eps if ec[e]==c]
        E=sum(sum(se(M[q],T[q]) for q in byep[e])/len(byep[e]) for e in es)/len(es); B=sum(sum(se(R[q],T[q]) for q in byep[e])/len(byep[e]) for e in es)/len(es); rat[c]=E/B; r+=E/B/4
    return 20/(1+r),rat
bt,ba=pts(A,tune)[0],pts(A,audit)[0]
print("IDEA 1: blend own/common, weight w on common (only short machines differ)")
for w in (.25,.5):
    M={q:[(1-w)*a+w*b for a,b in zip(A[q],Cm[q])] for q in T}
    print(f"  w={w}: tuning {pts(M,tune)[0]-bt:+.4f}  audit {pts(M,audit)[0]-ba:+.4f}")
# IDEA 2: are residuals correlated within episode? residual r = true_pwin - pred_pwin
print("IDEA 2: within-episode residual correlation of P(win) (does the episode err together?)")
for c in C:
    num=den=0; pairs=0
    for e in [e for e in byep if ec[e]==c]:
        rs=[(1-T[q][0])-(1-A[q][0]) for q in byep[e]]
        for i in range(len(rs)):
            for j in range(i+1,len(rs)): num+=rs[i]*rs[j]; pairs+=1
        den+=sum(x*x for x in rs)
    n=sum(len(byep[e]) for e in byep if ec[e]==c)
    print(f"  {c:14s} mean pairwise residual product / mean sq residual = {(num/pairs)/(den/n):+.3f}")
