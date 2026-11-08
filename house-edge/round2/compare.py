"""Collect whatever candidate scores have landed and print a comparison table + recommendation.
Safe to run repeatedly while jobs are still in flight (missing files are just skipped).

  python3 round2/compare.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

CANDIDATES = [
    ("thanos (plain E2)", os.path.join(R, "thanos_r2_score.json")),
    ("thanos + no-gate fix", os.path.join(R, "thanos_nogate_r2_score.json")),
    ("method4_ml base (v9)", os.path.join(R, "r2", "base", "final.json")),
    ("method4_ml sb (v9+shared-base)", os.path.join(R, "r2", "sb", "final.json")),
    ("adi.r2", os.path.join(R, "adi_r2_score.json")),
]

rows = []
for name, path in CANDIDATES:
    if not os.path.exists(path):
        print(f"[pending] {name}: no score file yet ({path})")
        continue
    d = json.load(open(path))
    tuning = d.get("tuning")
    audit = d.get("audit")
    full = d.get("full")
    rows.append((name, tuning, audit, full, d))

if not rows:
    print("\nNo candidates finished yet.")
else:
    print(f"\n{'candidate':34s} {'tuning':>8s} {'audit':>8s} {'full':>8s}")
    for name, t, a, f, d in rows:
        def fmt(x):
            return f"{x:8.4f}" if isinstance(x, (int, float)) else "     n/a"
        print(f"{name:34s} {fmt(t)} {fmt(a)} {fmt(f)}")
    best = max((r for r in rows if isinstance(r[3], (int, float))), key=lambda r: r[3], default=None)
    if best:
        print(f"\nBest by full validation: {best[0]} ({best[3]:.4f})")
    print("\nDecision rule (team bar, AGENTS.md rule 6 / CONTROL.md): adopt a change over the")
    print("fallback only if it gains >= 0.15 points on the AUDIT half (on this kit's 20-point scale,")
    print("that's the team's bar scaled down: ~0.05 pts) and the audit moves the same direction as")
    print("tuning, with no condition >10% worse. Prefer thanos unless method4_ml/adi.r2 clears that")
    print("bar, or thanos errors/fails the checker. Prefer thanos+no-gate-fix over plain thanos only")
    print("if it beats plain thanos on tuning AND audit (not just tuning) by more than noise (~48")
    print("tuning episodes per condition on this kit -- treat <0.02 pt full-score gaps as noise).")
