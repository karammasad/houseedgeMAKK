"""Inspect pinned Thanos screening mass from training histories; no answers read."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2_pool'))
from audit_thanos import GitCode
sys.meta_path.insert(0,GitCode('360ebc6'))
from common.data import public_dir,load_split
from thanos_method import egp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out');a=ap.parse_args()
    pub=public_dir(a.data);start=time.monotonic();q,s=load_split(pub,'train');ms=egp.machines_from_split(q,s)
    egp.TICKSET='own';keep,P=egp.screen(ms,min_mass=.01)
    groups=[]
    for i,g in enumerate(egp.ALL_GATES):
        mass=float(P[i].sum());held=[{'size_lag':j+1,'mass':float(p)} for j,p in enumerate(P[i]) if p>=.01]
        groups.append({'gate_lags':list(g),'pair_mass':mass,'retained_combinations':held,'discarded_mass':float(sum(p for p in P[i] if p<.01))})
    groups.sort(key=lambda x:-x['pair_mass']);result={'episodes':len({r['episode_id'] for r in q}),'machines':len(q),'threshold':.01,
        'retained_components':len(keep),'discarded_mass':sum(g['discarded_mass'] for g in groups),'gates':groups,
        'seconds':round(time.monotonic()-start,3),'warning':'Screen mass is not truth; omitted combinations cannot be predicted. Diagnostic only, no rule change.'}
    print(json.dumps(result,indent=2))
    if a.out:
        with open(a.out,'x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
