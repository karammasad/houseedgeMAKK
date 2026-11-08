import csv, collections, math
def xfeat(queries, hist, lags):
    """History-only cross-machine feature per query: mean over OTHER machines in the episode of
    [win rate at ticks cutoff-L (L in lags, observed only)] minus [that machine's overall observed win rate before cutoff]."""
    H=collections.defaultdict(dict)
    for r in hist: H[(r["episode_id"],r["machine_id"])][int(r["tick"])]=int(r["payout"])
    ep=collections.defaultdict(list)
    for q in queries: ep[q["episode_id"]].append(q)
    out={}
    for q in queries:
        vals=[]
        for o in ep[q["episode_id"]]:
            if o is q: continue
            h=H[(o["episode_id"],o["machine_id"])]; c=int(o["cutoff"])
            obs=[t for t in h if t<c]; base=sum(h[t]>0 for t in obs)/len(obs)
            rec=[h[c-L]>0 for L in lags if (c-L) in h]
            if rec: vals.append(sum(rec)/len(rec)-base)
        out[q["query_id"]]=sum(vals)/len(vals) if vals else 0.0
    return out
def adjust(p, f, beta):
    w=min(max(1-p[0],1e-9),1-1e-9); z=math.log(w/(1-w))+beta*f; w2=1/(1+math.exp(-z)); m=sum(p[1:])
    return [1-w2]+[x/m*w2 for x in p[1:]] if m>0 else list(p)
