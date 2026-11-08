"""Episode random effect on the win-rate logits (two-stage empirical Bayes at prediction time; uses fitted thanos hyperparameters).

Model: mu_on(m) = a_T + b_T s + u_on(e) + eps_m, u_on(e) ~ N(0, tau_on^2), eps ~ N(0, sd_T^2 - tau_on^2) (same for off). Stage 1: the
default thanos posterior of every machine gives a posterior mean r and variance v of its rate deviation; removing the N(0, sd_T^2) prior
gives a likelihood-only estimate l = r (1/v) / lam, precision lam = 1/v - 1/sd_T^2. Stage 2: for machine m, u(e) is estimated from the
OTHER machines of its episode only (leave-one-out: m's own data enters once, through its own likelihood), and m's posterior is recomputed
with prior N(a + b s + E[u | others], sigma^2 + Var[u | others]). tau^2 is a method-of-moments estimate from TRAIN machines' stage-1
likelihood estimates (within-episode pair covariance); no labels, targets or conditions are read.
"""
import math
from collections import defaultdict
import numpy as np
from . import egp


def stage1(machines, H):
    out = {}
    for m in machines:
        mo = {}; _, p = egp.machine_posterior(m, H, moments=mo)
        out[m.qid] = (p, mo.get("on"), mo.get("off"))
    return out


def lik(r, v, sd):
    lam = max(1.0 / v - 1.0 / sd ** 2, 1e-4); return r / v / lam, 1.0 / lam


def tau2(machines, H, S1):
    """Within-episode covariance of the likelihood estimates (pairs of different machines), floored at 0, per side."""
    res = {}
    for k, side in ((1, "on"), (2, "off")):
        by = defaultdict(list)
        for m in machines:
            x = S1[m.qid][k]
            if x is None:
                continue
            v = egp.unpack(H.theta[m.type])[0]; sd = math.exp(v["ls_" + side])
            l, n = lik(x[0], x[1], sd)
            if n < 4 * sd ** 2:                                   # ignore nearly uninformative machines
                by[m.ep].append(l)
        prods = [a * b for L in by.values() for i, a in enumerate(L) for b in L[i + 1:]]
        res[side] = max(float(np.mean(prods)), 0.0) if prods else 0.0
    return res


def stage2(machines, H, S1, T2, cap=0.8):
    by = defaultdict(list)
    for m in machines:
        by[m.ep].append(m)
    pred = {}
    for ep, ms in by.items():
        for m in ms:
            v = egp.unpack(H.theta[m.type])[0]; sh = []
            for k, side in ((1, "on"), (2, "off")):
                sd = math.exp(v["ls_" + side]); t2 = min(T2[side], cap * sd ** 2); s2 = sd ** 2 - t2
                P = 1.0 / max(t2, 1e-9); num = 0.0
                for j in ms:
                    x = S1[j.qid][k]
                    if j is m or x is None:
                        continue
                    sdj = math.exp(egp.unpack(H.theta[j.type])[0]["ls_" + side]); l, n = lik(x[0], x[1], sdj)
                    P += 1.0 / (s2 + n); num += l / (s2 + n)
                sh += [num / P, math.sqrt(s2 + 1.0 / P)]
            pred[m.qid] = egp.machine_posterior(m, H, shift=tuple(sh))[1].tolist()
    return pred


# ---------------------------------------------------------------------------------- correlated on / off rate prior (within machine)
def rho_hat(machines, H, S1, cap=0.8):
    """Per type, train histories only: posterior means r = k (delta + noise), k = 1 - v / s^2, so E[r_on r_off] = k_on k_off rho s_on s_off."""
    out = {}
    for t in sorted({m.type for m in machines}):
        v = egp.unpack(H.theta[t])[0]; so, sf = math.exp(v["ls_on"]), math.exp(v["ls_off"]); num = []; den = []
        for m in machines:
            x = S1[m.qid]
            if m.type != t or x[1] is None or x[2] is None:
                continue
            (ro, vo), (rf, vf) = x[1], x[2]
            num.append(ro * rf); den.append(max(1 - vo / so ** 2, 0) * max(1 - vf / sf ** 2, 0))
        out[t] = float(np.clip(np.mean(num) / max(np.mean(den), 1e-6) / (so * sf), 0.0, cap)) if num else 0.0
    return out


def corr_predict(machines, H, S1, RHO):
    pred = {}
    for m in machines:
        v = egp.unpack(H.theta[m.type])[0]; so, sf = math.exp(v["ls_on"]), math.exp(v["ls_off"]); r = RHO.get(m.type, 0.0)
        x = S1[m.qid]
        if r <= 0 or x[1] is None or x[2] is None:
            pred[m.qid] = x[0].tolist(); continue
        (ro, vo), (rf, vf) = x[1], x[2]
        sh = (r * so / sf * rf, math.sqrt(so ** 2 * (1 - r * r) + (r * so / sf) ** 2 * vf),
              r * sf / so * ro, math.sqrt(sf ** 2 * (1 - r * r) + (r * sf / so) ** 2 * vo))
        pred[m.qid] = egp.machine_posterior(m, H, shift=sh)[1].tolist()
    return pred
