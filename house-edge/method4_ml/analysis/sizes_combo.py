"""Joint GateSizes settings (tuning half; win prob fixed). python -m method4_ml.analysis.sizes_combo"""
import sys
sys.argv = [sys.argv[0]]
exec(open("method4_ml/analysis/sizes_grid.py").read().split("grid = json.loads")[0])
for over in [dict(tie_weight=0.9), dict(tie_weight=0.95), dict(tie_weight=0.9, couple_mix=0.03), dict(tie_weight=0.95, couple_mix=0.03),
             dict(tie_weight=0.9, couple_mix=0.03, couple_alpha=2.0), dict(tie_weight=0.9, couple_mix=0.05, couple_alpha=3.0, strength=8.0)]:
    tp, rr, sec = score(**over); print(over, f"tuning {tp:.3f}", {c: round(float(v), 4) for c, v in rr.items()}, flush=True)
