"""adi round 8: research-style stacker on the team FINAL (standard library). Owner: Adi.

Stacked generalization (Wolpert 1992; Ting & Witten 1997/1999): a low-capacity, L2-regularized logistic meta-learner
on the base model's probabilities and posterior summaries, trained on base predictions made without the label
(history ticks t, inputs from ticks < t). Builds on adi.sizewin (round 7: size-state terms) by adding gate posterior
summaries and, inspired by simultaneous-changepoint models (BASIC, Fan & Mackey 2015), the OTHER machines' flip
evidence in the same episode at the same tick.

  logit P'(win) = logit P_FINAL(win) + c . f

f (per machine, at tick t; FINAL's logit is a fixed offset):
  size  : d * P, d * (1 - P) per machine type  (d = size model's P(1x|win) - type mean)            [6, round 7]
  gate  : P(flip), P(flip) * (P_noflip - P), flip recency, P(no gate), short-log flag              [5]
  ep    : others' mean P(flip), others' max P(flip), others' mean (P - P_noflip), others' mean recency [4]
Fitted by Newton's method in pure Python on train history ticks (+ short-masked copies of whole episodes).

  python -m common.evaluate --method adi.stackr --data data/main
"""
import math, os, random
from multiprocessing import get_context
from common.data import load_split, wins
from adi import gate as G
from adi.model import is_short
from adi.sizewin.model import Model as SizeWin, TYPES, _logit

_G = {}
GROUPS = ("size", "gate", "ep")
WINDOWS = (10, 20, 80)


def cell_drift(win, r, s, P, W=40):
    """Gradual change inside the machine's current gate cell: under the most likely plain XOR pair (no flip), the
    win rate of ticks in the same cell as the cutoff over the last W ticks minus before. Returns [g, g*P] for long
    logs ([0, 0] for short logs or when the cell is unknown)."""
    if is_short(s):
        return [0.0, 0.0]
    w = wins(s); cut = len(w); best = None
    pairs = [st for f, st in win.ST if f == "xor" and len(st) == 2]
    tb = G.machine_tables(s, cut, [], [("xor", st) for st in pairs], .5, .5)
    i = max(range(len(pairs)), key=lambda k: tb[k][0][0]); a, b = pairs[i]
    if w[cut - a] < 0 or w[cut - b] < 0:
        return [0.0, 0.0]
    zc = w[cut - a] ^ w[cut - b]; rec = [0, 0]; old = [0, 0]
    for t in range(b, cut):
        if w[t] < 0 or w[t - a] < 0 or w[t - b] < 0 or (w[t - a] ^ w[t - b]) != zc:
            continue
        c = rec if t >= cut - W else old
        c[0] += 1; c[1] += w[t]
    if not rec[0] or not old[0]:
        return [0.0, 0.0]
    g = (rec[1] + .5) / (rec[0] + 1) - (old[1] + .5) / (old[0] + 1)
    return [g, g * P]


def multi_rate_feats(s, P):
    """Long logs: win rate over the last W observed ticks minus the rate before them, W in WINDOWS (gradual drift
    at several scales), alone and times P."""
    w = [x for x in wins(s) if x >= 0]; sh = float(is_short(s)); f = []
    for W in WINDOWS:
        rec, old = w[-W:], w[:-W]
        g = (sum(rec) + .5) / (len(rec) + 1) - (sum(old) + .5) / (len(old) + 1) if old else 0.0
        f += [(1 - sh) * g, (1 - sh) * g * P]
    return f


def rate_feats(s, P):
    """Win rates of the observed ticks in the last 40 vs before that. For short logs the older ones are the 24 sampled
    ticks the gate can't use (their lags are missing); for long logs the gap is a drift signal."""
    w = wins(s); cut = len(w); sh = float(is_short(s))
    rec = [x for x in w[cut - 40:] if x >= 0]; old = [x for x in w[:cut - 40] if x >= 0]
    r40 = (sum(rec) + .5) / (len(rec) + 1); ro = (sum(old) + .5) / (len(old) + 1)
    return [sh * (ro - r40), sh * (ro - r40) * P, (1 - sh) * (ro - r40), (1 - sh) * (ro - r40) * P]


def gate_summary(win, r, s):
    """(P, P_noflip, P(flip), recency/cut, P(no gate)) from the adi gate posterior at tick len(s)."""
    cfg = win._mcfg(s, r_type=r["machine_type"]); cut = len(s)
    taus, lp = G._taus(cut, cfg["p_change"], cfg["step"], cfg["tau_min"]); nt = len(taus)
    tb = G.machine_tables(s, cut, taus, win.ST, cfg["a"], cfg["b"], win._cp(r), cfg["half_life"])
    sp = win._sp(s); J = []
    for si, rows in enumerate(tb):
        if sp[si] <= 0:
            continue
        ls = math.log(sp[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append((ll + ls + lp[ti], pr, ti, si))
    z = G.logsumexp([j[0] for j in J]); P = Pnc = m_nc = rec = pnone = 0.0
    for l, pr, ti, si in J:
        w = math.exp(l - z); P += w * pr
        if ti == nt:
            m_nc += w; Pnc += w * pr
        else:
            rec += w * (cut - taus[ti])
        if si == len(tb) - 1:
            pnone += w
    pch = 1 - m_nc
    return P, (Pnc / m_nc if m_nc > 0 else P), pch, (rec / pch / cut if pch > 1e-9 else 1.0), pnone


def features(base, rs, sub, groups):
    """rs: an episode's query dicts with cutoff = t; sub: {machine_id: history truncated at t}.
    Returns {query_id: (offset logit, feature vector)}."""
    S = {r["machine_id"]: gate_summary(base.win, r, sub[r["machine_id"]]) for r in rs}
    out = {}
    for r in rs:
        mid = r["machine_id"]; P, Pnc, pch, rec, pn = S[mid]; f = []
        if "size" in groups:
            a = base.sizes.predict(r, sub[mid])[0]; d = a - base.sizes.mu.get(r["machine_type"], a)
            v = [0.0] * (2 * len(TYPES)); i = TYPES.index(r["machine_type"]); v[2 * i] = d * P; v[2 * i + 1] = d * (1 - P)
            f += v
        if "gate" in groups:
            f += [pch, pch * (Pnc - P), rec, pn, float(is_short(sub[mid]))]
        if "rate" in groups:
            f += rate_feats(sub[mid], P)
        if "cdrift" in groups:
            f += cell_drift(base.win, r, sub[mid], P)
        if "mrate" in groups:
            f += multi_rate_feats(sub[mid], P)
        if "ep" in groups:
            o = [S[x["machine_id"]] for x in rs if x["machine_id"] != mid] or [S[mid]]
            f += [sum(x[2] for x in o) / len(o), max(x[2] for x in o), sum(x[0] - x[1] for x in o) / len(o),
                  sum(x[3] for x in o) / len(o)]
        out[r["query_id"]] = (_logit(P), f)
    return out


def _job(args):
    rs, sub, t, tag = args
    rr = [dict(r, cutoff=t, query_id=f'{r["query_id"]}@{t}{tag}') for r in rs]
    return features(_G["base"], rr, {m: v[:t] for m, v in sub.items()}, _G["groups"])


class Model(SizeWin):
    def __init__(self, groups=GROUPS, **kw):
        super().__init__(**kw); self.groups = tuple(groups)

    def fit(self, pub):
        from method2_sizes.model import Model as M2
        self.base = M2(**self.base_kw).fit(pub)
        jobs, labels = [], []
        for split in self.splits:
            q, seq = load_split(pub, split); eps = {}
            for r in q:
                eps.setdefault(r["episode_id"], []).append(r)
            for e in sorted(eps):
                rs = eps[e]; cut = rs[0]["cutoff"]; sub = {r["machine_id"]: seq[r["machine_id"]] for r in rs}
                if any(min(v) < 0 for v in sub.values()):
                    continue
                rng = random.Random(f"{self.seed}|{e}")
                for t in sorted(rng.sample(range(self.t_min, cut), min(self.K, cut - self.t_min))):
                    jobs.append((rs, sub, t, ""))
                lo = max(self.t_min, 64)
                for t in sorted(rng.sample(range(lo, cut), min(self.K_short, cut - lo))):
                    msub = {}
                    for m, v in sub.items():
                        keep = set(range(t - 40, t)) | set(rng.sample(range(t - 40), 24))
                        msub[m] = [x if i in keep else -1 for i, x in enumerate(v[:t])] + v[t:]
                    jobs.append((rs, msub, t, "s"))
        _G["base"] = self.base; _G["groups"] = self.groups
        n = self.workers or min(os.cpu_count() or 1, 8)
        with get_context("fork").Pool(n) as pool:
            res = pool.map(_job, jobs, chunksize=4)
        X, y = [], []
        for (rs, sub, t, tag), feats in zip(jobs, res):
            for r in rs:
                lab = wins(sub[r["machine_id"]])[t]
                if lab >= 0:
                    off, f = feats[f'{r["query_id"]}@{t}{tag}']; X.append([1.0, off] + f); y.append(lab)
        self.n_train = len(X)
        self.coef = self._logreg(X, y)
        return self

    def predict(self, pub, split):
        base = self.base.predict(pub, split)
        q, seq = load_split(pub, split); eps = {}; out = {}
        for r in q:
            eps.setdefault(r["episode_id"], []).append(r)
        for rs in eps.values():
            F = features(self.base, rs, {r["machine_id"]: seq[r["machine_id"]] for r in rs}, self.groups)
            for r in rs:
                off, f = F[r["query_id"]]; x = [1.0, off] + f
                z = sum(b * v for b, v in zip(self.coef, x)); pw = 1 / (1 + math.exp(-max(min(z, 30), -30)))
                p = base[r["query_id"]]; pw0 = 1 - p[0]
                out[r["query_id"]] = [1 - pw] + [pw * v / pw0 for v in p[1:]] if pw0 > 0 else p
        return out
