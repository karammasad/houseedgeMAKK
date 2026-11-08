"""Exact sequence-level decoder for the smallest-lag configuration (xor lags (1,3), payout-size split at lag 1) -- ISOLATED EXPERIMENT.

Generative model (fixed, known parameters; this is the model the production gate integrates parameters over):
  categories  c_t in {0 loss, 1 = 1x win, 2 = big win}   (gate._cat3 of the payout);  w_t = [c_t > 0]
  ticks 0,1,2 : i.i.d. from the marginal m
  tick t >= 3 : gate bit g = w_{t-1} XOR w_{t-3}, polarity pol(t) = [t >= tau]; cell = (g XOR pol, c_{t-1});  r[g][c1] = win rate of the cell
                P(c_t = 0) = 1 - r, P(c_t = 1) = r (1 - beta), P(c_t = 2) = r beta
  tau         : none with prob 1 - pc, else uniform over {3, ..., u}  (u = target tick, a flip AT the target is allowed)
12-state sufficiency: the transition and emission at tick t read only (c_{t-1}, w_{t-2}, w_{t-3}) = 3 x 2 x 2 = 12 states; the next state is
(c_t, w_{t-1}, w_{t-2}).  Adding the 2 polarity values gives a 24-state chain with an exact forward recursion; a missing tick is summed over its
3 categories.  Machine-level parameters (r, beta) are held FIXED: that is what makes the chain finite (with unknown rates the latent counts
are exchangeable, not Markov, and no finite-state decoder exists).

exact_predict : forward algorithm over all ticks, missing ticks summed out.
local_predict : the production approximation under the same fixed parameters: tick t enters the likelihood only if c_t, c_{t-1}, w_{t-3}
                are all observed (gate._split_counts / gate_value drop every tick with a missing input); other ticks contribute nothing.
brute_predict : enumeration of every tau and every completion of the missing ticks (tiny sequences only).
"""
import itertools as it
import numpy as np

NS = 12


def sidx(c1, w2, w3):
    return c1 * 4 + w2 * 2 + w3


class Chain:
    def __init__(self, r, beta, m):
        self.r = np.asarray(r, float)            # (2 gate bit, 3 cat at lag 1)
        self.beta = float(beta); self.m = np.asarray(m, float)
        self.E = np.zeros((2, NS, 3)); self.nxt = np.zeros((NS, 3), int)
        for c1, w2, w3 in it.product(range(3), range(2), range(2)):
            s = sidx(c1, w2, w3); g = int(c1 > 0) ^ w3
            for pol in (0, 1):
                rr = self.r[g ^ pol, c1]
                self.E[pol, s] = (1 - rr, rr * (1 - beta), rr * beta)
            for k in range(3):
                self.nxt[s, k] = sidx(k, int(c1 > 0), w2)
        self.M = np.zeros((3, 2 * NS, 2 * NS))   # M[k][(pol,s) -> (pol,s')]
        for k in range(3):
            for pol in (0, 1):
                for s in range(NS):
                    self.M[k, pol * NS + s, pol * NS + self.nxt[s, k]] += self.E[pol, s, k]
        self.Msum = self.M.sum(axis=0)

    def p_cat(self, pol, c1, w2, w3):
        return self.E[pol, sidx(c1, w2, w3)]


def hazard(t, u, pc):
    """P(flip at tick t | no flip before t) for tau ~ {none: 1-pc, uniform on 3..u: pc}; t in 3..u."""
    n = u - 2
    surv = (1 - pc) + pc * (u - t + 1) / n
    return (pc / n) / surv


def _init_alpha(ch, obs):
    """Distribution over (pol=0, state at tick 3) after ticks 0..2 (observed or summed out); unnormalized joint prob."""
    a = np.zeros(2 * NS)
    for c in it.product(range(3), repeat=3):
        p = 1.0
        for t in range(3):
            if obs[t] >= 0 and obs[t] != c[t]:
                p = 0.0; break
            p *= ch.m[c[t]]
        if p:
            a[sidx(c[2], int(c[1] > 0), int(c[0] > 0))] += p
    return a


def exact_predict(ch, obs, u, pc):
    """obs: length-u list of categories (-1 = missing). Returns (P(c_u = k | obs) for k=0..2, joint probability of obs)."""
    a = _init_alpha(ch, obs)
    for t in range(3, u):
        h = hazard(t, u, pc)
        a = np.concatenate([a[:NS] * (1 - h), a[NS:] + a[:NS] * h])
        a = a @ (ch.Msum if obs[t] < 0 else ch.M[obs[t]])
    h = hazard(u, u, pc)
    a = np.concatenate([a[:NS] * (1 - h), a[NS:] + a[:NS] * h])
    joint = a.sum()
    pred = np.array([a[:NS] @ ch.E[0, :, k] + a[NS:] @ ch.E[1, :, k] for k in range(3)])
    return pred / joint, joint


def local_predict(ch, obs, u, pc):
    """Same fixed parameters; a tick counts only if c_t, c_{t-1}, w_{t-1}, w_{t-3} are observed. Target inputs must be observed."""
    a = np.array([1.0, 0.0])
    for t in range(3, u):
        h = hazard(t, u, pc)
        a = np.array([a[0] * (1 - h), a[1] + a[0] * h])
        if obs[t] >= 0 and obs[t - 1] >= 0 and obs[t - 3] >= 0:
            a = a * np.array([_emit(ch, p, obs, t) for p in (0, 1)])
    h = hazard(u, u, pc)
    a = np.array([a[0] * (1 - h), a[1] + a[0] * h]); a = a / a.sum()
    if min(obs[u - 1], obs[u - 3]) < 0:
        raise ValueError("target inputs missing")
    return sum(a[p] * _emit_dist(ch, p, obs, u) for p in (0, 1))


def _gate(ch, obs, t):
    return int(obs[t - 1] > 0) ^ int(obs[t - 3] > 0)


def _emit(ch, pol, obs, t):
    g = _gate(ch, obs, t); rr = ch.r[g ^ pol, obs[t - 1]]
    return (1 - rr, rr * (1 - ch.beta), rr * ch.beta)[obs[t]]


def _emit_dist(ch, pol, obs, t):
    g = _gate(ch, obs, t); rr = ch.r[g ^ pol, obs[t - 1]]
    return np.array([1 - rr, rr * (1 - ch.beta), rr * ch.beta])


def brute_predict(ch, obs, u, pc):
    """Enumerate tau in {None, 3..u} x all completions of the missing ticks. Returns (P(c_u=k | obs), joint)."""
    miss = [t for t in range(u) if obs[t] < 0]
    taus = [None] + list(range(3, u + 1))
    n = u - 2
    num = np.zeros(3); joint = 0.0
    for tau in taus:
        pt = (1 - pc) if tau is None else pc / n
        for comp in it.product(range(3), repeat=len(miss)):
            c = list(obs[:u]);
            for t, v in zip(miss, comp):
                c[t] = v
            p = pt
            for t in range(3):
                p *= ch.m[c[t]]
            for t in range(3, u):
                pol = 0 if (tau is None or t < tau) else 1
                p *= _emit(ch, pol, c, t)
            if p == 0:
                continue
            pol = 0 if (tau is None or u < tau) else 1
            g = _gate(ch, c + [0], u); rr = ch.r[g ^ pol, c[u - 1]]
            dist = np.array([1 - rr, rr * (1 - ch.beta), rr * ch.beta])
            joint += p; num += p * dist
    return num / joint, joint
