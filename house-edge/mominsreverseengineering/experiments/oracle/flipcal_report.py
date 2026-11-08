import pickle, zlib
import numpy as np
from sklearn.linear_model import LogisticRegression
R = [r for r in pickle.load(open("experiments/oracle/flipcal_rows.pkl", "rb")) if r["gated"]]; real = pickle.load(open("experiments/oracle/flipcal_real.pkl", "rb"))
sp = pickle.load(open("experiments/oracle/res1.pkl", "rb"))["size_p"]
F = np.array([r["F"] for r in R]); Pf = np.clip([r["Pf"] for r in R], 1e-6, 1 - 1e-6); short = np.array([r["short"] for r in R]); n = len(R)
s2 = np.array([1 + sum(x * x for x in sp[r["type"]]) for r in R]); pt = np.array([r["p_true"] for r in R])
E = lambda p: (np.array(p) - pt) ** 2 * s2 * 1e4
e0 = E([r["p"] for r in R]); et = E([r["ptau"] for r in R])
ef = E(np.where(F, [r["pf"] for r in R], [r["pn"] for r in R]))                      # knows flipped-or-not, not WHEN
eps = np.array([r["ep"] for r in R]); ue = sorted(set(eps)); EI = np.array([ue.index(e) for e in eps]); rng = np.random.default_rng(1)
def ci(d, sel=None):
    sel = np.ones(n, bool) if sel is None else sel
    ss = np.bincount(EI[sel], weights=d[sel], minlength=len(ue)); cc = np.bincount(EI[sel], minlength=len(ue)).astype(float); k = cc > 0; ss, cc = ss[k], cc[k]
    pick = rng.integers(0, len(ss), (2000, len(ss))); m = np.sort(ss[pick].sum(1) / cc[pick].sum(1)); return d[sel].mean(), m[50], m[1949]
f = lambda t: "%+.2f [%+.2f,%+.2f]" % t
print(f"{n} held-out simulated GATED machines (worlds 1,2; {int(short.sum())} short layout); {int(F.sum())} flipped before target ({F.mean():.2f})")
print("\n1. SIMULATOR vs REAL flip prevalence under the production posterior (576 real long train machines vs simulated long):")
sl = ~short
for nm, x in (("real long", real), ("sim long", Pf[sl])):
    print(f"   {nm:10s} mean P(flipped) {x.mean():.3f}   share >0.9: {np.mean(x > .9):.3f}   share <0.1: {np.mean(x < .1):.3f}   share 0.1-0.9: {np.mean((x >= .1) & (x <= .9)):.3f}")
print("\n2. CALIBRATION of P(flipped before target) (reliability bins; Wilson 95%)")
edges = [0, .02, .1, .3, .5, .7, .9, .98, 1.0001]; print(f"   {'bin':14s}{'n':>6s}{'mean P':>9s}{'observed':>10s}{'  95% CI':>16s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (Pf >= lo) & (Pf < hi); k = int(s.sum())
    if k == 0: continue
    ph = F[s].mean(); z = 1.96; c = (ph + z * z / (2 * k)) / (1 + z * z / k); h = z * np.sqrt(ph * (1 - ph) / k + z * z / (4 * k * k)) / (1 + z * z / k)
    print(f"   [{lo:.2f},{min(hi,1):.2f}){k:7d}{Pf[s].mean():9.3f}{ph:10.3f}   [{c-h:.3f},{c+h:.3f}]")
lg = np.log(Pf / (1 - Pf)).reshape(-1, 1); lr = LogisticRegression(C=1e6).fit(lg, F)
print(f"   calibration slope {lr.coef_[0][0]:.3f} (1 = calibrated), intercept {lr.intercept_[0]:+.3f} (0 = calibrated); Brier {np.mean((Pf-F)**2):.4f}, log-loss {-np.mean(np.where(F,np.log(Pf),np.log(1-Pf))):.4f}")
print("\n3. WHERE the flip time is (flipped machines): posterior mass within +-2 ticks of the true tau, 90% credible-set coverage and width")
fm = [r for r in R if r["F"] and r["cov"] is not None]
print(f"   n={len(fm)}  mean mass near true tau {np.mean([r['mass'] for r in fm]):.3f}  90% credible-set coverage {np.mean([r['cov'] for r in fm]):.3f}  median width {np.median([r['width'] for r in fm]):.0f} ticks")
for nm, sel in (("long", [not r["short"] for r in fm]), ("short", [r["short"] for r in fm])):
    ff = [r for r, k in zip(fm, sel) if k]; print(f"   {nm:6s} n={len(ff)} mass {np.mean([r['mass'] for r in ff]):.3f} coverage {np.mean([r['cov'] for r in ff]):.3f} median width {np.median([r['width'] for r in ff]):.0f}")
print("\n4. CONFIDENTLY WRONG vs DIFFUSE (call = flipped if P>0.5; confident = max(P,1-P) >= 0.9); five-category error x1e-4 vs truth")
conf = np.maximum(Pf, 1 - Pf) >= .9; call = Pf > .5; wrong = call != F
grp = {"confident, right": conf & ~wrong, "confident, WRONG": conf & wrong, "diffuse (<0.9)": ~conf}
tot = e0.sum()
print(f"   {'group':20s}{'n':>6s}{'mean err':>10s}{'share of total err':>20s}{'err if flip/no-flip known':>27s}{'err if tau known':>18s}")
for g, s in grp.items():
    print(f"   {g:20s}{int(s.sum()):6d}{e0[s].mean():10.2f}{e0[s].sum()/tot:20.3f}{ef[s].mean():27.2f}{et[s].mean():18.2f}")
print(f"   overall mean {e0.mean():.2f}; knowing flipped-or-not only: {ef.mean():.2f} ({f(ci(ef-e0))}); knowing tau: {et.mean():.2f}")
print("\n5. IS THERE A FIXABLE PROBLEM? cross-fitted recalibration of P(flipped) (Platt on logit, fit on one episode half, applied to the other)")
half = np.array([zlib.crc32(e.encode()) % 2 for e in eps]); P2 = np.zeros(n)
for h in (0, 1):
    m = LogisticRegression(C=1e6).fit(lg[half != h], F[half != h]); P2[half == h] = m.predict_proba(lg[half == h])[:, 1]
pf = np.array([r["pf"] for r in R]); pn = np.array([r["pn"] for r in R]); e2 = E(P2 * pf + (1 - P2) * pn)
print(f"   production {e0.mean():.2f}  recalibrated {e2.mean():.2f}   paired change {f(ci(e2-e0))}")
for nm, sel in (("long", ~short), ("short", short)): print(f"   {nm:6s} change {f(ci(e2-e0, sel))}")
print(f"   sanity: decomposition p = P*pf + (1-P)*pn reproduces production error: max |diff| {np.abs(E(Pf*pf+(1-Pf)*pn)-e0).max():.3f}")
