"""Training-only screen recovery diagnostic; no prediction targets or official audit data.

Run from mominsreverseengineering/: python round2_lag_research/screen_probe.py
Uses pinned team code read directly from git, without modifying or copying it.
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'round2_pool')); sys.path.insert(0,str(ROOT/'experiments'))
from audit_thanos import GitCode
sys.meta_path.insert(0,GitCode('360ebc6'))
from thanos_method import egp
from synth_kit import simulate, TYPES


def generate(seed, shifted, n_episodes=32):
    r=random.Random(seed); out=[]; actual=[]
    pairs=[(1,3),(2,5),(4,7),(3,6)]
    weights=[.34,.33,.33,0] if not shifted else [.40,.30,.25,.05]
    for ep in range(n_episodes):
        cut=r.randint(192,256); short=ep%4==0
        for j in range(8):
            typ=r.choice(list(TYPES)); signal=r.choice((-1,0,1)); pair=r.choices(pairs,weights)[0]
            history,_,_=simulate(r,typ,signal,pair,cut,None,.5,.34,.3,r.randrange(2))
            if short:
                keep=set(range(cut-40,cut)) | set(r.sample(range(cut-40),24))
                history=[x if i in keep else -1 for i,x in enumerate(history)]
            mid=f'e{ep}m{j}';out.append(egp.Machine(mid,f'e{ep}',mid,typ,signal,history));actual.append(pair)
    return out,actual


def main():
    t0=time.monotonic();results=[];egp.TICKSET='own'
    for shifted in (False,True):
        for seed in (73301,73302,73303,73304):
            ms,actual=generate(seed,shifted); keep,P=egp.screen(ms,min_mass=0.01)
            pair=(3,6); idx=egp.ALL_GATES.index(pair); entry=float(P[idx,2]); mass=float(P[idx,:].sum()); kept=[l for g,l,p in keep if g==pair]
            results.append(dict(scenario='shifted_5pct' if shifted else 'unchanged',seed=seed,episodes=32,
                actual_rare_count=actual.count(pair),screen_pair_mass=mass,screen_true_size_lag_mass=entry,
                kept_size_lags=kept,retains_rare_pair=bool(kept),retains_correct_combination=3 in kept,
                retained_components=len(keep),screen_discarded_mass=float(sum(P[a,b] for a in range(P.shape[0]) for b in range(P.shape[1]) if P[a,b]<.01))))
            print(json.dumps(results[-1]),flush=True)
    print('seconds',round(time.monotonic()-t0,2),flush=True)
    dest=ROOT/'round2_lag_research'/'screen_probe_results.json'
    with dest.open('w') as f:json.dump(results,f,indent=2)

if __name__=='__main__':main()
