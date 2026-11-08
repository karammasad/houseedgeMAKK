"""Verification of the win-informed size direction BEFORE any experiment. TRAIN/HISTORY data only; nothing here reads labels or targets."""
import copy, importlib.util, math, random, subprocess, sys
sys.path.insert(0, ".")
from common.data import public_dir, load_split
from adi import gate as G
from method4_ml.model import Model
from method4_ml.shared_base import win_direction_evidence
pub = public_dir("../data/main"); m = Model(cache=True, shared_base=True); gm = m._fit_gate(pub); g = m._fit_sizes(pub, gm)
# ---- T1: lambda = 0 reproduces the ORIGINAL (committed) size model exactly
src = subprocess.run(["git", "show", "6f8decd:house-edge/mominsreverseengineering/method2_sizes/sizes.py"], capture_output=True, text=True, cwd="..").stdout
spec = importlib.util.spec_from_loader("sizes_orig", loader=None); orig = importlib.util.module_from_spec(spec); exec(compile(src, "sizes_orig", "exec"), orig.__dict__)
o = orig.GateSizes.__new__(orig.GateSizes); o.__dict__.update(copy.deepcopy(g.__dict__))
g.win_dir_fn, g.win_dir_ctx = win_direction_evidence, gm
worst = {}
for lam in (0.0,):
    g.win_dir_lam = lam; w = 0.0; n = 0
    for split in ("validation", "test"):
        q, seq = load_split(pub, split)
        for r in q:
            s = seq[r["machine_id"]]; a = g.predict(r, s); b = o.predict(r, s); w = max(w, max(abs(x - y) for x, y in zip(a, b))); n += 1
    worst[lam] = (w, n)
print(f"T1 lambda=0 vs ORIGINAL committed size model: max |diff| = {worst[0.0][0]:.2e} over {worst[0.0][1]} queries (hook attached, lam=0)")
q, seq = load_split(pub, "validation"); g.win_dir_lam = 0.5
moved = max(max(abs(x - y) for x, y in zip(g.predict(r, seq[r["machine_id"]]), o.predict(r, seq[r["machine_id"]]))) for r in q[:200])
print(f"   sanity: with lam = 0.5 the predictions DO change (max |diff| on 200 queries = {moved:.3f}), so the hook is live when on")
g.win_dir_lam = 0.0
# ---- T2: vectorised per-direction win likelihood vs brute force
rng = random.Random(9); sig = lambda x: 1 / (1 + math.exp(-x)); grid = [-9 + 0.2 * i for i in range(81)]; SD = [[-1, -1, 1], [1, 1, -1]]
def brute(n, k, cp5):
    mu_on, mu_off, s_on, s_off, gam = cp5
    def w(mu, s):
        v = [math.exp(-0.5 * ((x - mu) / s) ** 2) for x in grid]; z = sum(v); return [y / z for y in v]
    won, woff = w(mu_on, s_on), w(mu_off, s_off); out = []
    for d in (0, 1):
        tot = 0.0
        for wts in ((won, woff), (woff, won)):
            z = 1.0
            for side in (0, 1):
                acc = 0.0
                for gi, x in enumerate(grid):
                    l = 1.0
                    for cat in range(3):
                        c = 3 * side + cat; th = x + gam * SD[d][cat]; l *= sig(th) ** k[c] * (1 - sig(th)) ** (n[c] - k[c])
                    acc += l * wts[side][gi]
                z *= acc
            tot += z
        out.append(math.log(tot / 2))
    return out
w2 = 0.0
for _ in range(40):
    cp5 = (rng.uniform(-1, 3), rng.uniform(-4, -1), rng.uniform(.2, .8), rng.uniform(.2, .8), rng.uniform(0, .8)); rows = []
    for _ in range(3):
        n = [rng.randint(0, 25) for _ in range(6)]; k = [rng.randint(0, x) for x in n]; rows.append((n, k, None))
    got = G._shared_dir_ll(rows, (14.0, 3.0, 0.2, 10.0, 0.0) + cp5)
    for r, (a0, a1) in zip(rows, got):
        b0, b1 = brute(r[0], r[1], cp5); w2 = max(w2, abs(a0 - b0), abs(a1 - b1))
        j = G._shared_tables([r], (14.0, 3.0, 0.2, 10.0, 0.0) + cp5)[0][0]; w2 = max(w2, abs(j - (math.log((math.exp(a0) + math.exp(a1)) / 2))))
print(f"T2 per-direction win likelihood vs brute force (and vs the existing joint _shared_tables): max |diff| = {w2:.2e} (120 rows)")
# ---- T3: product form == exact joint posterior over direction (no double counting), incl. a non-0.5 direction prior and a size likelihood
qv, sv = load_split(pub, "validation"); worst3 = 0.0; used = 0
for r in qv[:60]:
    s = sv[r["machine_id"]]
    for dl in (0.3, 0.5, 0.7):
        ev = win_direction_evidence(gm, r, s, dl)
        cfg = gm._mcfg(s, r_type=r["machine_type"]); taus, lp = G._taus(len(s), cfg["p_change"], cfg["step"], cfg["tau_min"]); cp = gm._cp(r); sp = gm._sp(s)
        for k, items in ev.items():
            num = [0.0, 0.0]
            for si, st in enumerate(gm.ST):
                if st[0] != "xsg" or st[1][2] != k or sp[si] <= 0:
                    continue
                rows = G.machine_counts(s, len(s), taus, [st], cfg.get("half_life"))[0][1]
                for ti, (l0, l1) in enumerate(G._shared_dir_ll(rows, cp)):
                    base = math.log(sp[si]) + lp[ti]; num[0] += (1 - dl) * math.exp(base + l0); num[1] += dl * math.exp(base + l1)
            exact_p1 = num[1] / (num[0] + num[1])                                       # P(d=1 | wins, k) from the exact joint
            lg = math.log(dl / (1 - dl)); mix_p1 = sum(w * sig(lg + llr) for w, llr in items)  # the hook's product form at lam = 1
            ell = (rng.uniform(-3, 3), rng.uniform(-3, 3))                              # an arbitrary size log-likelihood for d = 0, 1
            ex = [num[0] * math.exp(ell[0]), num[1] * math.exp(ell[1])]; ex = ex[1] / (ex[0] + ex[1])
            pf = [(1 - mix_p1) * math.exp(ell[0]), mix_p1 * math.exp(ell[1])]; pf = pf[1] / (pf[0] + pf[1])
            worst3 = max(worst3, abs(exact_p1 - mix_p1), abs(ex - pf)); used += 1
print(f"T3 product form vs exact joint posterior over direction: max |diff| = {worst3:.2e} over {used} (machine, size lag, direction prior) cases")
