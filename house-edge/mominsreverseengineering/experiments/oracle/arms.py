"""Oracle arms on one simulated world (fit the REAL gate pipeline on the synthetic train set, then predict fresh synthetic test machines).
All numbers are P(win at the target tick); error is measured against the TRUE probability (no outcome noise)."""
import copy, math, os, pickle, sys, time
import numpy as np
sys.path.insert(0, "experiments/oracle")
import sim
from adi import gate as G
from multiprocessing import get_context

ARMS = ["A0 production", "Pc true cell-prior params", "Ps true structure prior+family", "Pp true population (=Bayes B0)",
        "A0+struct known", "A0+tau known", "B0+struct known", "B0+tau known", "B0+struct+tau known", "B0+rates known"]
_S = {}


def truth_gm(gm, world):
    """gm with the TRUE population: xsg structures only, true cell-prior parameters, true structure prior, true flip prior."""
    g = copy.copy(gm); pop = world["pop"]; pri = dict(gm.cellpriors)
    for t in sim.TYPES:
        for sg in (-1, 0, 1):
            base = tuple(gm.cellpriors[(t, sg)][:5]); pri[(t, sg)] = base + tuple(pop[t]["cp"][sg])
    g.cellpriors = pri; P = pop[sim.TYPES[0]]
    g.ST = [("xsg", pr) for pr in P["pairs"]]; g.sprior = list(P["pair_p"]); g.p_change_short = 0.5
    return g


def pw(gm, m, sprior=None, tau=None, ST=None):
    r = dict(machine_type=m["type"], signal=m["signal"]); ss = m["hist"]; L = m["L"]; cfg = gm._mcfg(ss, r_type=m["type"])
    st = ST or gm.ST; sp = sprior if sprior is not None else gm._sp(ss)
    if tau == "true":
        if m["tau"] is None: taus, lp = [], [0.0]
        else: taus, lp = [m["tau"]], [0.0, -1e300]
    else:
        taus, lp = G._taus(L, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(ss, L, taus, st, cfg["a"], cfg["b"], gm._cp(r), cfg["half_life"])
    return G._combine(tb, sp, lp)[0]


def pw_rates(gm_t, m):
    ss = m["hist"]; L = m["L"]; cfg = gm_t._mcfg(ss, r_type=m["type"]); taus, lp = G._taus(L, cfg["p_change"], cfg["step"], cfg["tau_min"])
    C = G.machine_counts(ss, L, taus, gm_t.ST, None); rt = np.clip(m["rates"].reshape(6), 1e-9, 1 - 1e-9); J = []; PR = []
    for si, (tag, rows) in enumerate(C[:-1]):
        ls = math.log(gm_t.sprior[si])
        for ti, (nz, kz, zc) in enumerate(rows):
            nz = np.array(nz, float); kz = np.array(kz, float)
            J.append(float((kz * np.log(rt) + (nz - kz) * np.log(1 - rt)).sum()) + ls + lp[ti]); PR.append(float(rt[zc]))
    n0, k0 = C[-1]; rb = float(np.clip(rt.mean(), 1e-9, 1 - 1e-9)); J.append(k0 * math.log(rb) + (n0 - k0) * math.log(1 - rb) + math.log(gm_t.sprior[-1]) + lp[0]); PR.append(rb)
    J = np.array(J); w = np.exp(J - J.max()); w /= w.sum(); return float((w * np.array(PR)).sum())


def true_sprior(gm, m, st):
    sp = [0.0] * (len(st) + 1)
    if m["pair"] is None: sp[-1] = 1.0
    else: sp[st.index(("xsg", m["pair"]))] = 1.0
    return sp


def job(m):
    g0, gt, world = _S["g0"], _S["gt"], _S["world"]; out = []
    P = world["pop"][m["type"]]
    sp_tf = [0.0] * (len(g0.ST) + 1)                                              # true structure prior inside A0's family
    for pr, p in zip(P["pairs"], P["pair_p"][:3]): sp_tf[g0.ST.index(("xsg", pr))] = p
    sp_tf[-1] = P["pair_p"][3]
    g_pc = copy.copy(g0); pri = dict(g0.cellpriors)
    for sg in (-1, 0, 1): pri[(m["type"], sg)] = tuple(g0.cellpriors[(m["type"], sg)][:5]) + tuple(P["cp"][sg])
    g_pc.cellpriors = pri
    out.append(pw(g0, m)); out.append(pw(g_pc, m)); out.append(pw(g0, m, sprior=sp_tf)); out.append(pw(gt, m))
    out.append(pw(g0, m, sprior=true_sprior(g0, m, g0.ST))); out.append(pw(g0, m, tau="true"))
    out.append(pw(gt, m, sprior=true_sprior(gt, m, gt.ST))); out.append(pw(gt, m, tau="true"))
    out.append(pw(gt, m, sprior=true_sprior(gt, m, gt.ST), tau="true")); out.append(pw_rates(gt, m))
    return out


def run_arms(gm, world, procs=2):
    _S.update(g0=gm, gt=truth_gm(gm, world), world=world)
    with get_context("fork").Pool(procs) as p: res = p.map(job, world["test"], chunksize=20)
    P = np.array(res)                                                              # (n_test, n_arms)
    meta = [dict(type=m["type"], short=m["short"], flipped=m["tau"] is not None, gated=m["pair"] is not None, ep=m["ep"], p_true=m["p_true"], pair=m["pair"]) for m in world["test"]]
    return P, meta


if __name__ == "__main__":
    seed = int(sys.argv[1]); procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    from fitworld import fit_world
    ntest = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    w = sim.make_world(seed, n_test=ntest); wd = f"experiments/oracle/work{seed}"; os.makedirs(wd, exist_ok=True)
    gp = os.path.join(wd, "gm.pkl")
    if os.path.exists(gp): gm = pickle.load(open(gp, "rb")); tm = {}
    else: gm, tm = fit_world(w, wd); pickle.dump(gm, open(gp, "wb"))
    t0 = time.time(); P, meta = run_arms(gm, w, procs)
    pickle.dump(dict(P=P, meta=meta, size_p=w["size_p"], fit_times=tm, arm_time=time.time() - t0), open(f"experiments/oracle/res{seed}.pkl", "wb"))
    print("seed", seed, "done", tm, round(time.time() - t0, 1))
