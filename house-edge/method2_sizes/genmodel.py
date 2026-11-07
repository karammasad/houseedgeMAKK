"""Hierarchical generative model of a machine's history, built from the fitted FINAL parameters, plus the
observable statistics used for a posterior predictive check (PPC). Standard library only.

Generative story (what our fitted models jointly assume):
  type, signal                         given (taken from a real query template)
  win structure s ~ adi structure prior (XOR single / pair; split forms act as their pair); polarity d ~ Bern(1/2)
  flip: with prob p_flip the XOR side swaps at tau ~ U[tau_min, cut - 10]; in a `shared` episode all machines share tau
  on/off win rates ~ Beta from adi's per-type, signal-shifted cell priors (machine-level random effects)
  win bit w_t ~ Bern(rate[XOR(w_{t-a}, w_{t-b}) ^ d ^ flipped])   (generated tick by tick from earlier outcomes)
  size lag k ~ P(size lag | win structure) (method 2's learned coupling table); tied w.p. tie_weight:
      partition p ~ part_prior, direction ~ Bern(1/2); state_t = group of category(payout_{t-k})
  P(1x | win) = levels[type][state_t]; else 2x/5x/20x ~ split[type]
  optional `state_win_shift`: logit shift of the win rate by size state (the link adi found in round 4)
  observation mask: copied from the template machine (so short logs keep their 40 + 24 sampled-tick layout)
"""
import itertools, math, random

SIZES = (1, 2, 5, 20)
PARTITIONS = ((2,), (1,), (0,))


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + math.exp(-x))


def cellprior(par, typ, sig):
    mo, mf, ko, kf, so, sf = par["signal_priors"][typ]
    a = _sig(_logit(mo) + so * sig); b = _sig(_logit(mf) + sf * sig)
    return a * ko, (1 - a) * ko, b * kf, (1 - b) * kf


class Simulator:
    def __init__(self, par, p_flip=0.4, shared_frac=0.25, state_win_shift=None, seed=0):
        self.par, self.p_flip, self.shared_frac, self.shift = par, p_flip, shared_frac, state_win_shift or {}
        self.rng = random.Random(seed)
        self.ST = [(s[0], tuple(s[1])) for s in par["ST"]] + [("none", ())]
        self.sp = par["sprior"]
        self.wST = [tuple(s) for s in par["wST"]]

    def _lag_row(self, lags):
        key = tuple(lags)
        return self.par["cond"][self.wST.index(key)] if key in self.wST else self.par["cond"][len(self.wST)]

    def machine(self, typ, sig, cut, mask, tau_shared=None, tau=-1, with_truth=False):
        """tau: -1 = legacy random flip (p_flip); None = no flip; int = flip at that tick."""
        R, par = self.rng, self.par
        form, lags = R.choices(self.ST, self.sp)[0]
        if form not in ("xor", "xs", "xst") or not lags:
            form, lags = "none", ()
        elif form in ("xs", "xst"):
            lags = tuple(lags[:2])   # adi stores (i, j, split_lag); only (i, j) enter the XOR (a repeated lag would cancel)
        d = R.randrange(2); a_on, b_on, a_off, b_off = cellprior(par, typ, sig)
        r_on, r_off = R.betavariate(a_on, b_on), R.betavariate(a_off, b_off)
        if tau == -1:
            tau = tau_shared if tau_shared is not None else (R.randint(40, max(41, cut - 10)) if R.random() < self.p_flip else None)
        k = R.choices(range(9), self._lag_row(lags))[0]
        lo, hi = par["levels"][typ]; lv = (lo, hi); split = par["split"][typ]
        tied = R.random() < min(1.0, par["tie_weight"]) if isinstance(par["tie_weight"], float) else True
        grp = PARTITIONS[R.choices(range(3), par["part_prior"])[0]]; dirn = R.randrange(2)
        const_state = R.randrange(2); free_state = [R.randrange(2) for _ in range(3)]
        shift = self.shift.get(typ, 0.0)
        s = [0] * cut; w = [0] * cut; mw = max(lags) if lags else 0
        for t in range(cut):
            if k and t - k >= 0:
                c = 0 if s[t - k] == 0 else 1 if s[t - k] == 1 else 2
                st = (dirn if c in grp else 1 - dirn) if tied else free_state[c]
            else:
                st = const_state if k == 0 else R.randrange(2)
            if form == "none" or t < mw:
                rate = r_on if R.random() < .5 else r_off
            else:
                x = 0
                for l in lags:
                    x ^= w[t - l]
                x ^= d ^ (1 if (tau is not None and t >= tau) else 0)
                rate = r_on if x == 1 else r_off
            if shift:
                rate = _sig(_logit(rate) + shift * (1 if st == 1 else -1))
            w[t] = 1 if R.random() < rate else 0
            if w[t]:
                s[t] = 1 if R.random() < lv[st] else R.choices((2, 5, 20), split)[0]
        seq_out = [v if m else -1 for v, m in zip(s, mask)]
        if not with_truth:
            return seq_out
        # true next-spin distribution at tick `cut` (exactly the generative rules above, applied one more step)
        t = cut
        if k and t - k >= 0:
            c = 0 if s[t - k] == 0 else 1 if s[t - k] == 1 else 2
            st = (dirn if c in grp else 1 - dirn) if tied else free_state[c]
        else:
            st = const_state if k == 0 else 0.5
        if form == "none":
            rate = .5 * (r_on + r_off)
        else:
            x = 0
            for l in lags:
                x ^= w[t - l]
            x ^= d ^ (1 if (tau is not None and t >= tau) else 0)
            rate = r_on if x == 1 else r_off
        def lvl_at(state):
            return lv[state]
        def rate_at(state):
            return _sig(_logit(rate) + shift * (1 if state == 1 else -1)) if shift else rate
        if st == 0.5:   # unknown state: average the two states
            pw = .5 * (rate_at(0) + rate_at(1)); a = (.5 * rate_at(0) * lv[0] + .5 * rate_at(1) * lv[1]) / pw
        else:
            pw = rate_at(st); a = lv[st]
        truth = [1 - pw, pw * a] + [pw * (1 - a) * z for z in split]
        return seq_out, truth


# ------------------------------------------------------------------ observable statistics
PAIRS = list(itertools.combinations(range(1, 9), 2))


def _frac(k, n):
    return k / n if n else None


def machine_stats(seq):
    """Scalar/vector observables of one (possibly masked) history; ticks with value -1 are unobserved."""
    cut = len(seq); w = [-1 if v < 0 else int(v > 0) for v in seq]
    obs = [t for t in range(cut) if w[t] >= 0]
    out = {"nobs": len(obs), "rate": sum(w[t] for t in obs) / max(1, len(obs))}
    wins = [v for v in seq if v > 0]
    out["n_wins"] = len(wins)
    for z in SIZES:
        out[f"share_{z}"] = _frac(sum(1 for v in wins if v == z), len(wins))
    # best XOR pair
    best = None; eff = {}
    for a, b in PAIRS:
        n = [0, 0]; k = [0, 0]
        for t in range(b, cut):
            if w[t] >= 0 and w[t - a] >= 0 and w[t - b] >= 0:
                x = w[t - a] ^ w[t - b]; n[x] += 1; k[x] += w[t]
        if min(n) >= 8:
            e = k[1] / n[1] - k[0] / n[0]; eff[(a, b)] = e
            if best is None or abs(e) > abs(best[1]):
                best = ((a, b), e, k[1] / n[1], k[0] / n[0])
    out["best_e"] = abs(best[1]) if best else None
    out["hi_rate"] = max(best[2], best[3]) if best else None
    out["lo_rate"] = min(best[2], best[3]) if best else None
    # half-split sign consistency of the best pair
    out["flip"] = None
    if best and len(obs) > 150:
        a, b = best[0]; mid = obs[len(obs) // 2]; es = []
        for lo_, hi_ in ((0, mid), (mid, cut)):
            n = [0, 0]; k = [0, 0]
            for t in range(max(b, lo_), hi_):
                if w[t] >= 0 and w[t - a] >= 0 and w[t - b] >= 0:
                    x = w[t - a] ^ w[t - b]; n[x] += 1; k[x] += w[t]
            es.append(k[1] / n[1] - k[0] / n[0] if min(n) >= 8 else None)
        if None not in es and min(abs(es[0]), abs(es[1])) > .25:
            out["flip"] = 1.0 if es[0] * es[1] < 0 else 0.0
    # single-lag win effects (mean |effect| over lags 1..8)
    se = []
    for l in range(1, 9):
        n = [0, 0]; k = [0, 0]
        for t in range(l, cut):
            if w[t] >= 0 and w[t - l] >= 0:
                n[w[t - l]] += 1; k[w[t - l]] += w[t]
        if min(n) >= 8:
            se.append(abs(k[1] / n[1] - k[0] / n[0]))
    out["single_e"] = sum(se) / len(se) if se else None
    # size-lag effect: P(1x | win, big at lag k) - P(1x | win, not big)
    bestk = None
    for l in range(1, 9):
        n = [0, 0]; k = [0, 0]
        for t in range(l, cut):
            if seq[t] > 0 and seq[t - l] >= 0:
                g = int(seq[t - l] > 1); n[g] += 1; k[g] += int(seq[t] == 1)
        if min(n) >= 8:
            e = k[1] / n[1] - k[0] / n[0]
            if bestk is None or abs(e) > abs(bestk[1]):
                bestk = (l, e)
    out["size_e"] = abs(bestk[1]) if bestk else None
    # win rate inside the on-cell by size state (big at the size lag) -> size-win link, sign-adjusted
    out["link"] = None
    if best and bestk and abs(bestk[1]) > .2:
        (a, b), e, r1, r0 = best; on = 1 if r1 > r0 else 0; l = bestk[0]; n = [0, 0]; k = [0, 0]
        for t in range(max(b, l), cut):
            if w[t] >= 0 and w[t - a] >= 0 and w[t - b] >= 0 and seq[t - l] >= 0 and (w[t - a] ^ w[t - b]) == on:
                g = int(seq[t - l] > 1); n[g] += 1; k[g] += w[t]
        if min(n) >= 8:
            sgn = 1 if bestk[1] > 0 else -1     # big@lag raises P(1x) when sgn > 0
            out["link"] = (k[1] / n[1] - k[0] / n[0]) * sgn
    # run lengths
    runs = {0: [], 1: []}; cur = None; ln = 0
    for t in range(cut):
        if w[t] < 0:
            cur = None; continue
        if w[t] == cur:
            ln += 1
        else:
            if cur is not None:
                runs[cur].append(ln)
            cur, ln = w[t], 1
    out["win_run"] = sum(runs[1]) / len(runs[1]) if runs[1] else None
    out["loss_run"] = sum(runs[0]) / len(runs[0]) if runs[0] else None
    return out


def is_short(seq):
    return sum(v < 0 for v in seq) > len(seq) / 2
