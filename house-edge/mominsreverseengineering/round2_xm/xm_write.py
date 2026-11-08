"""python xm_write.py <public> <split> <thanos_csv> <out_csv> [beta] : apply cross-machine adjustment A (lags 1..8) to a Thanos CSV. History-only."""
import sys, csv; sys.path.insert(0,"/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-rEZWCR/e0c65641-1400-4a02-8422-550c7f4089b5/scratchpad/r2work")
from xm import xfeat, adjust
pub,sp,inp,out=sys.argv[1:5]; beta=float(sys.argv[5]) if len(sys.argv)>5 else 2.0
qs=list(csv.DictReader(open(f"{pub}/{sp}/queries.csv"))); hs=list(csv.DictReader(open(f"{pub}/{sp}/history.csv")))
F=xfeat(qs,hs,range(1,9)); P=list(csv.DictReader(open(inp)))
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["query_id","p_0","p_1","p_2","p_5","p_20"])
    for r in P:
        p=adjust([float(r[c]) for c in ("p_0","p_1","p_2","p_5","p_20")],F[r["query_id"]],beta); s=sum(p); w.writerow([r["query_id"]]+[repr(x/s) for x in p])
print("wrote",out,len(P))
