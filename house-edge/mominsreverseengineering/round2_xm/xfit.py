import sys; sys.path.insert(0,"/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-rEZWCR/e0c65641-1400-4a02-8422-550c7f4089b5/scratchpad/r2work")
exec(open("/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-rEZWCR/e0c65641-1400-4a02-8422-550c7f4089b5/scratchpad/r2work/idea.py").read().split("bt,ba=")[0])
from xm import xfeat, adjust
import random
qs=list(csv.DictReader(open(pub+"/validation/queries.csv"))); hs=list(csv.DictReader(open(pub+"/validation/history.csv")))
F=xfeat(qs,hs,range(1,9))
betas=[0.25,0.5,1,1.5,2,2.5,3,4]
def cur(eps): b=pts(A,eps)[0]; return {be:pts({q:adjust(A[q],F[q],be) for q in T},eps)[0]-b for be in betas}
ct,ca=cur(tune),cur(audit)
print("beta    tune     audit   (gain vs Thanos)"); [print(f" {be:4}  {ct[be]:+.4f}  {ca[be]:+.4f}") for be in betas]
print("cross-fit: best beta on tuning =",max(ct,key=ct.get),"-> audit",f"{ca[max(ct,key=ct.get)]:+.4f}","| best beta on audit =",max(ca,key=ca.get),"-> tuning",f"{ct[max(ca,key=ca.get)]:+.4f}")
# 20 random episode half-splits (stratified by condition): fit beta on one half, score the other
random.seed(5); outs=[]
eps_c={c:[e for e in byep if ec[e]==c] for c in C}
for i in range(20):
    h1=set()
    for c in C: es=eps_c[c][:]; random.shuffle(es); h1|=set(es[:len(es)//2])
    h2=set(byep)-h1; g1=cur(h1); bb=max(g1,key=g1.get); g2=cur(h2); outs.append((bb,g2[bb]))
print("20 random half-splits: fitted beta",sorted(b for b,_ in outs)," held-out gain min %+.4f median %+.4f max %+.4f, positive in %d/20"%(min(g for _,g in outs),sorted(g for _,g in outs)[10],max(g for _,g in outs),sum(g>0 for _,g in outs)))
# per-type and per-condition win counts at beta 2
M={q:adjust(A[q],F[q],2.0) for q in T}
for t in ("balanced","jackpot","steady"):
    s=[q for q in T if Q[q]["machine_type"]==t]; print(f"  type {t:8s} err x1e-4 {1e4*sum(se(A[q],T[q]) for q in s)/len(s):.2f} -> {1e4*sum(se(M[q],T[q]) for q in s)/len(s):.2f}")
