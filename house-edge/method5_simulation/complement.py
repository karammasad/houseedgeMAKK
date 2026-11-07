"""Does Method 5 hold information the strongest team model lacks? (diagnostic only; tuning half; audit untouched)

  python -m method5_simulation.complement CACHE.json --data data/main
CACHE.json = cached validation predictions of team models (key 'v5' = Kareem's stacker, 58.19 full; 'final' = Method 2).
Candidates: B2 and S2 (frozen, trained on train only; out/validation_*.csv). Reported on the validation TUNING half:
  1. standalone competition points and condition ratios (exact five-category formula)
  2. per-episode squared error vs the TRUE probabilities: team vs candidate, correlation, share of episodes won
  3. correlation of the error vectors e = P - P_true (all 5 components, all queries; per condition)
  4. incremental value: regress the team's remaining error (P_true - P_team) on the disagreement (P_cand - P_team);
     beta = optimal blend weight, with an episode-clustered bootstrap CI. beta ~ 0 => nothing the team model lacks.
  5. the competition score of a few FIXED blends (predeclared weights; not tuned)
"""
import argparse, csv, json, os
import numpy as np
from common.data import public_dir
from common.evaluate import folds, points, COLS, CONDITIONS


def read(path):
    return {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); tune, _ = folds(pub); rng = np.random.default_rng(0)
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}; cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    ep = {r["query_id"]: r["episode_id"] for r in rd("queries.csv")}
    C = json.load(open(a.cache)); team = {"v5 (Kareem; team best available)": {k: np.array(v) / sum(v) for k, v in C["v5"]["validation"].items()},
                                         "Method 2 final": {k: np.array(v) / sum(v) for k, v in C["final"]["validation"].items()}}
    cand = {"B2": read("method5_simulation/out/validation_b2.csv"), "S2": read("method5_simulation/out/validation_s2.csv"),
            "S1 (ablation ref.)": read("method5_simulation/out/validation_s1.csv")}
    ids = [q for q in T if ep[q] in tune]; eps = sorted({ep[q] for q in ids}); E = np.array([ep[q] for q in ids]); K = np.array([cond[q] for q in ids])
    tr = np.array([T[q] for q in ids])
    print("1. standalone competition score, validation tuning half (organizer baseline = 30)")
    print(f"   {'model':40s} {'tuning':>8s}  " + "  ".join(f"{c[:6]:>6s}" for c in CONDITIONS))
    for n, P in list(team.items()) + list(cand.items()):
        tp, r = points(pub, {q: list(map(float, P[q])) for q in T}, tune); print(f"   {n:40s} {tp:8.3f}  " + "  ".join(f"{r[c]:.4f}" for c in CONDITIONS))
    for tname, TP in team.items():
        Pt = np.array([TP[q] for q in ids]); et = Pt - tr
        print(f"\n=== reference team model: {tname} ===")
        for cname, CP in cand.items():
            Pc = np.array([CP[q] for q in ids]); ec = Pc - tr
            per_t = np.array([(et[E == e] ** 2).sum(1).mean() for e in eps]); per_c = np.array([(ec[E == e] ** 2).sum(1).mean() for e in eps])
            corr_e = np.corrcoef(et.ravel(), ec.ravel())[0, 1]
            cc = "  ".join(f"{c[:5]} {np.corrcoef(et[K == c].ravel(), ec[K == c].ravel())[0, 1]:+.2f}" for c in CONDITIONS)
            d = (Pc - Pt).ravel(); y = (tr - Pt).ravel(); beta = (d @ y) / (d @ d)
            def beta_of(sel):
                dd = (Pc[sel] - Pt[sel]).ravel(); yy = (tr[sel] - Pt[sel]).ravel(); return (dd @ yy) / (dd @ dd)
            bs = []
            for _ in range(500):
                pick = rng.choice(len(eps), len(eps)); sel = np.concatenate([np.nonzero(E == eps[i])[0] for i in pick]); bs.append(beta_of(sel))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"-- {cname}: per-episode error {per_c.mean():.5f} vs team {per_t.mean():.5f} (candidate wins {np.mean(per_c < per_t):.0%} of {len(eps)} episodes; "
                  f"episode-error correlation {np.corrcoef(per_t, per_c)[0, 1]:+.2f})")
            print(f"   error-vector correlation {corr_e:+.3f} overall; per condition: {cc}")
            print(f"   incremental value: beta = {beta:+.3f} [{lo:+.3f}, {hi:+.3f}]  (optimal blend weight on the candidate; CI excludes 0 only if it adds information)")
            base, _ = points(pub, {q: list(map(float, TP[q])) for q in T}, tune); line = []
            for w in (0.05, 0.10, 0.25, 0.50):
                B = {q: list(map(float, (1 - w) * TP[q] + w * np.array(CP[q]))) for q in T}; line.append(f"w={w:.2f}: {points(pub, B, tune)[0] - base:+.3f}")
            print(f"   fixed blends, change in tuning score vs team alone: " + "   ".join(line))


if __name__ == "__main__":
    main()
