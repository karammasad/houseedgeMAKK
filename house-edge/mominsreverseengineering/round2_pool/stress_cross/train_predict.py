"""Generate train-cutoff forecasts from a fitted model; no train labels read."""
import os,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
with open(sys.argv[1],'rb') as f:m=pickle.load(f)
m.workers=2; m.cache=False
p=m.predict(os.path.abspath(sys.argv[2]),'train')
with open(sys.argv[3],'wb') as f:pickle.dump(p,f)
print('Predicted',len(p),'training cutoffs')
