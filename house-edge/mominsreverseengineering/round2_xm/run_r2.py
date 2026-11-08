"""python run_r2.py <public> <outdir> <0|1 nogate> <splits: val|val,test>. One thread. Thanos copy unmodified; fix applied by monkeypatch only."""
import os, sys, time, json, math, hashlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
S="/private/tmp/claude-501/-Users-mominhassan--claude-spares-spare-rEZWCR/e0c65641-1400-4a02-8422-550c7f4089b5/scratchpad/r2work"
sys.path.insert(0,S+"/thanos_copy"); sys.path.insert(1,"/Users/mominhassan/houseedgeMAKK/house-edge/mominsreverseengineering/experiments/rehearsal")
from common.data import public_dir, write_predictions
from thanos import egp
from thanos.model import Model
import nogate_fix
pub=public_dir(sys.argv[1]); out=sys.argv[2]; fix=sys.argv[3]=="1"; sp=sys.argv[4].split(","); os.makedirs(out,exist_ok=True)
if fix: nogate_fix.apply(egp)
t0=time.time(); M=Model("all",verbose=True,tickset="own").fit(pub); t1=time.time()
info=dict(fix=fix,fit_s=round(t1-t0,1))
for s in sp:
    name={"val":"validation","test":"test"}[s]; t=time.time(); P=M.predict(pub,name); write_predictions(os.path.join(out,name+".csv"),P); info["predict_"+name]=round(time.time()-t,1)
H=M.H; info["K"]=len(H.K); info["nogate_mass"]=[round(float(math.exp(H.lpi[i])),4) for i,(G,l) in enumerate(H.K) if len(G)==0]
info["support"]=[(list(G),l,round(float(math.exp(H.lpi[i])),3)) for i,(G,l) in enumerate(H.K)]
info["rho"]={str(k):round(v,3) for k,v in H.rho.items()}
info["wall_s"]=round(time.time()-t0,1); json.dump(info,open(out+"/run.json","w"),indent=1); print(info)
