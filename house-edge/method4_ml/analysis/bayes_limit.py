"""Posterior predictive check: is the gate's actual P(win) error ~ its own posterior variance (Bayes limit under the model)?
implied Var = sum_c w_c [ within_c + (p_c - pbar)^2 ], within_c ~ p_c(1-p_c)/(n_cell + k + 1) (Beta posterior variance).
If actual MSE >> implied Var in a slice, the model is misspecified there (room to improve); if ~equal, the error is
irreducible GIVEN the model's assumptions. Diagnosis only (validation truth)."""
import csv, math
from collections import defaultdict
from common.data import load_split
from method4_ml.model import Model
from adi import gate as G
PUB = "data/main"
def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
T = {r["query_id"]: 1 - float(r["p_0"]) for r in rows(f"{PUB}/validation/targets.csv")}
C = {r["query_id"]: r["condition"] for r in rows(f"{PUB}/validation/conditions.csv")}
gm = Model(blend=0).fit(PUB).gm; q, seq = load_split(PUB, "validation")
agg = defaultdict(lambda: [0, 0.0, 0.0])
for r in q:
    s = seq[r["machine_id"]]; cfg = gm._mcfg(s, r_type=r["machine_type"]); cp = gm._cp(r); sp = gm._sp(s)
    taus, lprior = G._taus(r["cutoff"], cfg["p_change"], cfg["step"], cfg["tau_min"])
    Cn = G.machine_counts(s, r["cutoff"], taus, gm.ST, cfg.get("half_life"))
    tb = G.tables_from_counts(Cn, len(taus) + 1, cfg["a"], cfg["b"], cp)
    k = ((cp[0] + cp[1]) + (cp[2] + cp[3])) / 2 if cp else 1.0
    J, PR, W = [], [], []
    for si, (rows_t, rows_c) in enumerate(zip(tb, Cn)):
        if sp[si] <= 0: continue
        for ti, (ll, pr) in enumerate(rows_t):
            if si == len(tb) - 1:
                n = sum(1 for x in s if x >= 0)
            else:
                nz, kz, zc = rows_c[ti]; n = nz[zc] if zc is not None else 0
            J.append(ll + math.log(sp[si]) + lprior[ti]); PR.append(pr); W.append(pr * (1 - pr) / (n + k + 1))
    z = G.logsumexp(J); w = [math.exp(j - z) for j in J]; pbar = sum(a * b for a, b in zip(w, PR))
    var = sum(a * (wv + (p - pbar) ** 2) for a, wv, p in zip(w, W, PR))
    for key in ("ALL", C[r["query_id"]], r["machine_type"]):
        x = agg[key]; x[0] += 1; x[1] += (pbar - T[r["query_id"]]) ** 2; x[2] += var
print(f"{'slice':14s} {'n':>4s} {'actual MSE':>11s} {'implied var':>12s} {'ratio':>6s}")
for key in ("ALL", "stable", "short_history", "drift", "shared_change", "steady", "balanced", "jackpot"):
    n, a, v = agg[key]; print(f"{key:14s} {n:4d} {a / n:11.5f} {v / n:12.5f} {a / v:6.2f}")
