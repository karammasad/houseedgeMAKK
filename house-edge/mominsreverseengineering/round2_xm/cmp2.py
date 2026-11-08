"""python cmp2.py base.csv cand.csv : official metric (20 pts) on full/tuning/audit halves (kit split: per condition, sha256-sorted episodes alternating),
per-condition ratio change, short-history jackpot subset, per-episode drivers, episode bootstrap. Scoring only."""
import csv, sys, collections, hashlib, random
pub="/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-rEZWCR/e0c65641-1400-4a02-8422-550c7f4089b5/scratchpad/r2kit/public"
rd=lambda p:{r["query_id"]:[float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")] for r in csv.DictReader(open(p))}
A,B,T,R=rd(sys.argv[1]),rd(sys.argv[2]),rd(pub+"/validation/targets.csv"),rd(pub+"/validation/reference.csv")
cond={r["query_id"]:r["condition"] for r in csv.DictReader(open(pub+"/validation/conditions.csv"))}
Q={r["query_id"]:r for r in csv.DictReader(open(pub+"/validation/queries.csv"))}
n=collections.Counter()
for r in csv.DictReader(open(pub+"/validation/history.csv")): n[(r["episode_id"],r["machine_id"])]+=1
short=lambda q:n[(Q[q]["episode_id"],Q[q]["machine_id"])]<0.5*int(Q[q]["cutoff"])
C=["stable","short_history","drift","shared_change"]
se=lambda p,t:sum((a-b)**2 for a,b in zip(p,t))
byep=collections.defaultdict(list)
for q in T: byep[Q[q]["episode_id"]].append(q)
ec={e:cond[v[0]] for e,v in byep.items()}
tune,audit=set(),set()
for c in C:
    o=sorted([e for e in byep if ec[e]==c],key=lambda e:hashlib.sha256(e.encode()).hexdigest()); tune|=set(o[::2]); audit|=set(o[1::2])
em=lambda M,e:sum(se(M[q],T[q]) for q in byep[e])/len(byep[e])
def score(M,eps):
    E={c:[em(M,e) for e in eps if ec[e]==c] for c in C}; E={c:sum(v)/len(v) for c,v in E.items()}
    Er={c:[em(R,e) for e in eps if ec[e]==c] for c in C}; Er={c:sum(v)/len(v) for c,v in Er.items()}
    rat={c:E[c]/Er[c] for c in C}; Rm=sum(rat.values())/4; return 20/(1+Rm),rat
for nm,eps in (("FULL",set(byep)),("TUNING",tune),("AUDIT",audit)):
    pa,ra=score(A,eps); pb,rb=score(B,eps)
    print(f"{nm:6s} eps={len(eps)} base {pa:.4f} cand {pb:.4f} diff {pb-pa:+.4f}")
    for c in C: print(f"     {c:14s} ratio {ra[c]:.5f} -> {rb[c]:.5f}  ({100*(rb[c]/ra[c]-1):+.1f}%)")
# bootstrap on full, within condition
random.seed(917); eps_c={c:[e for e in byep if ec[e]==c] for c in C}; Er={c:sum(em(R,e) for e in eps_c[c])/len(eps_c[c]) for c in C}
da={e:em(A,e) for e in byep}; db={e:em(B,e) for e in byep}
def pts(d,es): return 20/(1+sum(sum(d[e] for e in es[c])/len(es[c])/Er[c] for c in C)/4)
bs=[]
for _ in range(3000):
    es={c:[random.choice(eps_c[c]) for _ in eps_c[c]] for c in C}; bs.append(pts(db,es)-pts(da,es))
bs.sort(); print(f"FULL episode bootstrap diff mean {sum(bs)/len(bs):+.4f} 95% [{bs[75]:+.4f}, {bs[2924]:+.4f}] P(cand better) {sum(x>0 for x in bs)/len(bs):.3f}")
# subsets (mean sq err x1e-4)
def sub(nm,f):
    s=[q for q in T if f(q)]; a=1e4*sum(se(A[q],T[q]) for q in s)/len(s); b=1e4*sum(se(B[q],T[q]) for q in s)/len(s)
    better=sum(se(B[q],T[q])<se(A[q],T[q])-1e-12 for q in s); worse=sum(se(B[q],T[q])>se(A[q],T[q])+1e-12 for q in s)
    print(f"  {nm:34s} n={len(s):4d} base {a:7.2f} cand {b:7.2f} ({100*(b/a-1):+.1f}%)  queries better/worse {better}/{worse}")
print("subsets (x1e-4 mean sq err):")
sub("short_history jackpot",lambda q:cond[q]=="short_history" and Q[q]["machine_type"]=="jackpot")
sub("short_history jackpot TUNING",lambda q:cond[q]=="short_history" and Q[q]["machine_type"]=="jackpot" and Q[q]["episode_id"] in tune)
sub("short_history jackpot AUDIT",lambda q:cond[q]=="short_history" and Q[q]["machine_type"]=="jackpot" and Q[q]["episode_id"] in audit)
for t in ("balanced","steady"): sub("short_history "+t,lambda q,t=t:cond[q]=="short_history" and Q[q]["machine_type"]==t)
sub("short-log machines (any cond)",short); sub("long-log machines",lambda q:not short(q))
# episode drivers: contribution to ratio delta
d={e:(db[e]-da[e])/Er[ec[e]]/len(eps_c[ec[e]])/4 for e in byep}   # contribution to delta R
tot=sum(d.values()); print(f"delta R total {tot:+.6f} (neg = cand better)")
srt=sorted(d.items(),key=lambda x:x[1])
print(" top-5 helping episodes:",[(e[-6:],ec[e][:6],"T" if e in tune else "A",round(v/tot,2)) for e,v in srt[:5]])
print(" top-5 hurting episodes:",[(e[-6:],ec[e][:6],"T" if e in tune else "A",round(v/tot,2)) for e,v in srt[-5:]])
print(f" episodes better/worse/same: {sum(v<-1e-12 for v in d.values())}/{sum(v>1e-12 for v in d.values())}/{sum(abs(v)<=1e-12 for v in d.values())}")
k=5; print(f" delta R without top-{k} helping episodes: {tot-sum(v for _,v in srt[:k]):+.6f}")
