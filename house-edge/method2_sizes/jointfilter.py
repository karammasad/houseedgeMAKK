"""Joint win x size hidden-state marginalization (Method 2, experiment 13). Standard library only.

Extends missfilter.py. A hidden lag tick is ONE payout category c in {loss, 1x, big}: it determines the win bit that
feeds the gate (c > 0) AND the size state that feeds the size model (c in the learned group, e.g. big). So for a tick
whose lags are unobserved we marginalize that single hidden category jointly for both outcomes:

  P(w_t, size_t | r_on, r_off, d, o) = sum over hidden categories of  P(cats) * Bern(w_t; rate(x)) * Bern(size_t; level(state))
      x = [cat_a > 0] XOR [cat_b > 0],   rate = r_on if x == o else r_off,   state = d if cat_k in group else 1 - d

so a sampled WIN's size (1x vs big) tells us about the hidden lag category, hence the hidden win bit and gate cell.
Separate models (a win filter + a size model) cannot use this coupling.

Implementation: for each tick pattern the hidden-category enumeration is collapsed ONCE into weights W[x][g]
(g = cat_k in group), independent of the rates; the grid loop then costs 4 multiplications per pattern. Known-input
ticks are the one-term special case, so a single code path covers everything. Hidden categories use the machine's own
win rate p and 1x share; missing ticks are never treated as losses; ticks are never assumed adjacent.

Hypotheses: gate pairs kept by the structure prior learned in missfilter.MissFilter (prior >= prune), orientation
o and size direction d in {0,1}, size lag k = argmax of method 2's learned coupling row for the pair (a parameter).
Levels per type and the partition come from method 2's fitted GateSizes (numbers only). Rates integrated on a G x G
grid against the type's Beta priors. Output: posterior-mean 5-vector at the cutoff with and without unknown-input
ticks; the difference is the information existing models discard.
"""
import itertools, math
from collections import defaultdict
from method2_sizes.missfilter import PAIRS, lse

GROUPS = ((2,), (1,), (0,))   # partition 0: {big}, 1: {1x}, 2: {loss}


def cat(v):
    return 0 if v == 0 else 1 if v == 1 else 2


class JointFilter:
    def __init__(self, mf, par, G=12, prune=0.01):
        self.mf, self.par, self.G = mf, par, G
        self.g, self.lg, self.l1g = mf.g, mf.lg, mf.l1g
        self.wST = [tuple(s) for s in par["wST"]]; self.cond = par["cond"]
        pp = par["part_prior"]; self.grp = GROUPS[max(range(3), key=pp.__getitem__)]
        names = [(p, o) for p in PAIRS for o in (0, 1)]
        keep = [(names[i], mf.prior[i]) for i in range(len(names)) if mf.prior[i] >= prune]
        z = sum(p for _, p in keep); self.hyps = [(n, p / z) for n, p in keep]

    def klag(self, pair):
        key = tuple(pair)
        if key not in self.wST:
            return pair[0]
        row = self.cond[self.wST.index(key)]; k = max(range(len(row)), key=row.__getitem__)
        return pair[0] if k == 0 else k

    # ---- tick patterns ------------------------------------------------------------------------------------------------
    def patterns(self, s, pair, k):
        """Group observed ticks by (categories at the distinct lags or None if hidden, win, size outcome)."""
        U = sorted({pair[0], pair[1], k}); known, unk = defaultdict(int), defaultdict(int)
        for t in range(max(U), len(s)):
            if s[t] < 0:
                continue
            cs = tuple(cat(s[t - l]) if s[t - l] >= 0 else None for l in U)
            w = int(s[t] > 0); sz = (1 if s[t] == 1 else 2) if s[t] > 0 else None
            (unk if None in cs else known)[(cs, w, sz)] += 1
        return U, known, unk

    def weights(self, U, cs, pair, k, p, s1):
        """W[x][g]: probability of (gate-XOR x, size-group indicator g) with hidden categories summed out."""
        qc = (1 - p, p * s1, p * (1 - s1)); hid = [i for i, c in enumerate(cs) if c is None]
        W = [[0.0, 0.0], [0.0, 0.0]]; ia, ib, ik = U.index(pair[0]), U.index(pair[1]), U.index(k)
        for vals in itertools.product((0, 1, 2), repeat=len(hid)):
            c = list(cs); pr = 1.0
            for i, v in zip(hid, vals):
                c[i] = v; pr *= qc[v]
            W[int(c[ia] > 0) ^ int(c[ib] > 0)][1 if c[ik] in self.grp else 0] += pr
        return W

    # ---- evidence on the rate grid ------------------------------------------------------------------------------------
    def evidence(self, r, s, pair, k, o, d, with_unknown=True, want_cut=True):
        """Return (list of log weights over the G x G grid including the Beta priors, patterns) for one hypothesis."""
        mf, G = self.mf, self.G; typ = r["machine_type"]; lv = self.par["levels"][typ]
        w = [-1 if v < 0 else int(v > 0) for v in s]; obs = [x for x in w if x >= 0]
        p = min(.95, max(.05, sum(obs) / max(1, len(obs)))); wins = [v for v in s if v > 0]
        s1 = min(.95, max(.05, (sum(1 for v in wins if v == 1) + 1) / (len(wins) + 2)))
        U, kn, un = self.patterns(s, pair, k); pr_on, pr_off = mf._grid_prior(typ, r["signal"])
        pats = [(key, c) for key, c in kn.items()] + ([(key, c) for key, c in un.items()] if with_unknown else [])
        pre = []
        for (cs, wv, sz), c in pats:
            W = self.weights(U, cs, pair, k, p, s1)
            # F = sum_{x,g} W[x][g] * rateLik(x) * sizeLik(g)    (rate: on-cell iff x == o; state: d if g else 1 - d)
            terms = []
            for x in (0, 1):
                for g in (0, 1):
                    if W[x][g] > 0:
                        st = d if g else 1 - d
                        sl = 1.0 if sz is None else (lv[st] if sz == 1 else 1 - lv[st])
                        terms.append((W[x][g] * sl, x == o, wv))
            pre.append((c, terms))
        out = []
        for i in range(G):
            gi = self.g[i]
            for j in range(G):
                gj = self.g[j]; ll = pr_on[i] + pr_off[j]
                for c, terms in pre:
                    f = 0.0
                    for wgt, on, wv in terms:
                        rt = gi if on else gj; f += wgt * (rt if wv else 1 - rt)
                    ll += c * math.log(max(1e-300, f))
                out.append(ll)
        return out, (U, p, s1)

    def run(self, r, s, with_unknown=True):
        """Posterior-mean 5-vector [p0, p1, p2, p5, p20] at the cutoff; None if the cutoff cell/state is unobserved."""
        typ = r["machine_type"]; lv = self.par["levels"][typ]; split = self.par["split"][typ]; G = self.G
        cut = len(s); terms = []
        for (pair, o), pi in self.hyps:
            k = self.klag(pair); lags = {pair[0], pair[1], k}
            if any(s[cut - l] < 0 for l in lags):
                return None
            xc = int(s[cut - pair[0]] > 0) ^ int(s[cut - pair[1]] > 0); on = (xc == o); gk = cat(s[cut - k]) in self.grp
            for d in (0, 1):
                ll, _ = self.evidence(r, s, pair, k, o, d, with_unknown)
                lvl = lv[d if gk else 1 - d]
                for idx, x in enumerate(ll):
                    i, j = divmod(idx, G); terms.append((x + math.log(pi * .5), self.g[i] if on else self.g[j], lvl))
        z = lse([t[0] for t in terms]); er = erl = 0.0
        for ll, rate, lvl in terms:
            wgt = math.exp(ll - z); er += wgt * rate; erl += wgt * rate * lvl
        return [1 - er, erl] + [(er - erl) * x for x in split]
