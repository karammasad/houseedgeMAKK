"""How much of v8's remaining error is reducible through better inference from short histories? TRAIN HISTORIES ONLY.

No validation file, no audit, no targets. Targets are REALIZED win outcomes (0/1) of later ticks of long train machines; a paired
difference of squared errors against a realized outcome is an unbiased estimate of the difference of squared errors against the true
probability (E[y] = p cancels the cross terms), but absolute levels include the irreducible Bernoulli noise p(1-p). Never read an
absolute Brier here as "error against the truth".

Pseudo-query: machine m, target tick t in [45%, 58%) of its log (polarity constant there with high probability), prediction uses ONLY
ticks < t (inputs of the target cell come from the 7 ticks before t, observed in a real short log too). Information settings, all
relative to t:  n10 / n20 / n40 / n80 = the n ticks immediately before t;  short64 = real short layout (40 consecutive + 24 random older);
full = every tick before t (the more-informed benchmark; never future ticks).
Predictors, identical prefixes and targets:
  V0   v8's production gate (cached; flips, 6-cell size split, Beta cell priors per type x signal)
  HA   hierarchical grid predictor, independent Beta base rates, gamma (shared base rate + state shift, direction averaged)
  HC   same, bivariate logit-normal with on/off correlation rho          HD   HC with sigma and signal slopes shared across types
  V0o  V0 with the structure fixed to the full-history MAP pair ("structure known")   HCo  the same for HC
Population priors of HA/HC/HD are CROSS-FITTED: episode-grouped 4 folds, a fold's machines are predicted by priors fit on the other 3.
(V0's prior was fit on all train histories, i.e. it has a mild in-sample advantage: conservative for the alternatives.)
Run:  python3 prefixdiag.py --data ../data/main
"""
import argparse, math, os, time, zlib, json, random
from collections import defaultdict
from multiprocessing import get_context
import numpy as np
from common.data import public_dir, load_split
from adi import gate as G
import hierarchy as H

MASKS = ("n10", "n20", "n40", "n80", "short64", "full")
MODELS = {"HA": dict(prior="beta"), "HC": dict(prior="ln", rho=True), "HD": dict(prior="ln", rho=True, share=("ls_on", "ls_off", "b_on", "b_off"))}
NF = 4
_S = {}


def fold_of(ep):
    return zlib.crc32(ep.encode()) % NF


def fit_task(args):
    fold, name = args
    d = _S["data"]; rows = d[0]
    idx = np.array([i for i, r in enumerate(rows) if fold is None or fold_of(r["ep"]) != fold])
    sub = ([rows[i] for i in idx], d[1][idx], d[2][idx], d[3][idx], d[4][idx])
    h = H.Hier(sub, **MODELS[name]); h.fit(maxiter=120)
    return fold, name, h.x.tolist(), h.names, h.ll, len(idx)


def keep_set(mask, t, s, mid):
    if mask == "full":
        return None
    if mask.startswith("n"):
        n = int(mask[1:]); return set(range(max(0, t - n), t))
    rng = random.Random(zlib.crc32(f"{mid}|{t}".encode())); older = list(range(0, t - 40))
    return set(range(t - 40, t)) | set(rng.sample(older, min(24, len(older))))


def masked(s, t, keep):
    return list(s[:t]) if keep is None else [v if i in keep else -1 for i, v in enumerate(s[:t])]


def v0_predict(gm, r, t, ss, restrict=None):
    cfg = gm._mcfg(ss, r_type=r["machine_type"]); taus, lprior = G._taus(t, cfg["p_change"], cfg["step"], cfg["tau_min"])
    tb = G.machine_tables(ss, t, taus, gm.ST, cfg["a"], cfg["b"], gm._cp(r), cfg["half_life"]); sp = gm._sp(ss)
    J, PR, KEY = [], [], []
    for si, rows in enumerate(tb):
        if sp[si] <= 0:
            continue
        key = "none" if si == len(gm.ST) else gm.ST[si][1][:2]
        if restrict is not None and key != restrict:
            continue
        ls = math.log(sp[si])
        for ti, (ll, pr) in enumerate(rows):
            J.append(ll + ls + lprior[ti]); PR.append(pr); KEY.append(key)
    z = G.logsumexp(J); w = [math.exp(j - z) for j in J]
    mass = defaultdict(float)
    for wi, k in zip(w, KEY):
        mass[k] += wi
    return sum(wi * p for wi, p in zip(w, PR)), mass


def targets_for(s, t):
    w = [-1 if v < 0 else int(v > 0) for v in s]; out = []
    for (i, j) in H.PAIRS:
        out.append((w[t - i] ^ w[t - j], 0 if s[t - i] == 0 else (1 if s[t - i] == 1 else 2)))
    return out


def machine_job(args):
    r, = args; s = _S["seq"][r["machine_id"]]; L = len(s); fold = fold_of(r["episode_id"]); gm = _S["gm"]; rows = []
    P = _S["params"]
    for t in range(int(.45 * L), int(.58 * L), 2):
        y = int(s[t] > 0); targ = targets_for(s, t); full_pair = None; full_post = {}
        res = {}
        for mask in ("full",) + tuple(m for m in MASKS if m != "full"):
            ss = masked(s, t, keep_set(mask, t, s, r["machine_id"]))
            K, N, kt, nt = H.extract(ss, t)
            p0, mass = v0_predict(gm, r, t, ss)
            rec = dict(ep=r["episode_id"], mid=r["machine_id"], type=r["machine_type"], sig=r["signal"], t=t, mask=mask, y=y, V0=p0,
                       ident=max(mass.values()))
            for name in MODELS:
                h, x = P[(fold, name)]
                ph, post = h.predict(x, r["machine_type"], r["signal"], K, N, kt, nt, targ); rec[name] = ph
                if name == "HC":
                    rec["_postHC"] = post
            if mask == "full":
                full_pair = max(mass, key=mass.get)
                hc_idx = int(np.argmax(rec["_postHC"]))
                full_hc_post = rec["_postHC"]
            else:
                p0o, _ = v0_predict(gm, r, t, ss, restrict=full_pair); rec["V0o"] = p0o
                h, x = P[(fold, "HC")]
                # HC with the structure known: renormalise its posterior to the full-history MAP structure
                K2, N2 = K.copy(), N.copy(); post = rec["_postHC"]
                rec["HCo"] = _hc_oracle(h, x, r, K, N, kt, nt, targ, hc_idx)
            rec.pop("_postHC", None); rows.append(rec)
        # full mask rows get oracle = themselves
        for rec in rows[-len(MASKS):]:
            if rec["mask"] == "full":
                rec["V0o"] = rec["V0"]; rec["HCo"] = rec["HC"]
    return rows


def _hc_oracle(h, x, r, K, N, kt, nt, targ, idx):
    """HC prediction with the structure fixed to index idx (0..2 = pair, 3 = none)."""
    G_ = H
    gam = h.get(x, "gam", r["machine_type"]); L = H.side_loglik(K[None], N[None], gam)
    if idx == 3:
        mu, sd = h.get(x, "mu_n", r["machine_type"]), math.exp(h.get(x, "ls_n", r["machine_type"])); wn = np.exp(-0.5 * ((H.G - mu) / sd) ** 2); wn /= wn.sum()
        Ln = kt * H.LOGSIG + (nt - kt) * H.LOG1MSIG; post = np.exp(Ln - Ln.max()) * wn
        return float((post / post.sum() / 1 * 1 / (1 + np.exp(-H.G))).sum() * 1)
    c = np.maximum(L[0].max(-1), L[1].max(-1))[0]; E = [np.exp(L[d][0] - c[..., None]) for d in (0, 1)]; W = h.W(x, r["machine_type"], r["signal"])
    parity, cat = targ[idx]; lz = []; rr = []
    for d in (0, 1):
        e0, e1 = E[d][idx, 0], E[d][idx, 1]
        for assign in (0, 1):
            eon, eoff = (e0, e1) if assign == 0 else (e1, e0)
            Pon = eon * (W @ eoff); Poff = eoff * (W.T @ eon); Z = Pon.sum()
            Pr = Pon if (parity == 0) == (assign == 0) else Poff
            lz.append(math.log(Z + 1e-300)); rr.append(float((Pr / Z / (1 + np.exp(-(H.G + gam * H.SD[d][cat])))).sum()))
    a = np.array(lz); w = np.exp(a - a.max()); return float((w * np.array(rr)).sum() / w.sum())


def boot(delta, eps, reps=1000, seed=3):
    by = defaultdict(list)
    for d, e in zip(delta, eps):
        by[e].append(d)
    keys = sorted(by); rng = random.Random(seed); ms = []
    sums = {k: (sum(v), len(v)) for k, v in by.items()}
    for _ in range(reps):
        ch = [sums[rng.choice(keys)] for _ in keys]; ms.append(sum(a for a, _ in ch) / sum(b for _, b in ch))
    ms.sort(); return ms[int(.025 * reps)], ms[int(.975 * reps)]


def report(rows):
    def mean(v):
        return sum(v) / len(v)
    def brier(rs, k):
        return mean([(r[k] - r["y"]) ** 2 for r in rs])
    print(f"\npseudo-queries: {len({(r['mid'], r['t']) for r in rows})} (machine, tick) pairs from {len({r['mid'] for r in rows})} long machines, "
          f"{len({r['ep'] for r in rows})} episodes")
    print("\nMEAN Brier vs REALIZED outcome (x1e-4; includes the irreducible Bernoulli floor) and PAIRED difference vs V0 (x1e-4, 95% CI by episode):")
    print(f"{'info':8s} {'n':>6s} {'V0':>8s} | " + " | ".join(f"{k:>22s}" for k in ("HA", "HC", "HD", "V0o (struct known)", "HCo")))
    for mask in MASKS:
        rs = [r for r in rows if r["mask"] == mask]; line = f"{mask:8s} {len(rs):6d} {brier(rs, 'V0') * 1e4:8.1f} | "
        cells = []
        for k in ("HA", "HC", "HD", "V0o", "HCo"):
            d = [((r[k] - r["y"]) ** 2 - (r["V0"] - r["y"]) ** 2) * 1e4 for r in rs]; lo, hi = boot(d, [r["ep"] for r in rs])
            cells.append(f"{mean(d):+6.2f} [{lo:+6.2f},{hi:+6.2f}]")
        print(line + " | ".join(cells))
    print("\nBy machine type, short64 information (paired difference vs V0, x1e-4):")
    for ty in ("balanced", "jackpot", "steady"):
        rs = [r for r in rows if r["mask"] == "short64" and r["type"] == ty]
        print(f"  {ty:9s} n={len(rs):4d} V0 Brier {brier(rs, 'V0') * 1e4:7.1f}   " + "   ".join(
            f"{k} {mean([((r[k] - r['y']) ** 2 - (r['V0'] - r['y']) ** 2) * 1e4 for r in rs]):+6.2f}" for k in ("HA", "HC", "HD", "V0o")))
    print("\nBy gate identifiability under V0 (max structure-pair posterior >= .9), short64 and n40:")
    for mask in ("short64", "n40"):
        for lab, sel in (("identified", lambda r: r["ident"] >= .9), ("ambiguous", lambda r: r["ident"] < .9)):
            rs = [r for r in rows if r["mask"] == mask and sel(r)]
            if len(rs) < 20:
                print(f"  {mask:7s} {lab:10s} n={len(rs)} (too few)"); continue
            print(f"  {mask:7s} {lab:10s} n={len(rs):4d} V0 {brier(rs, 'V0') * 1e4:7.1f}  " + "  ".join(
                f"{k} {mean([((r[k] - r['y']) ** 2 - (r['V0'] - r['y']) ** 2) * 1e4 for r in rs]):+6.2f}" for k in ("HA", "HC", "HD", "V0o")))
    full = brier([r for r in rows if r["mask"] == "full"], "V0"); sh = brier([r for r in rows if r["mask"] == "short64"], "V0")
    ok = brier([r for r in rows if r["mask"] == "short64"], "V0o")
    print(f"\nDECOMPOSITION of V0's short64 Brier (x1e-4; the floor is shared by all rows): short64 {sh * 1e4:.1f}  |  full-history {full * 1e4:.1f}"
          f"  |  short64 with structure known {ok * 1e4:.1f}")
    print(f"  more data (short64 -> full):   {(sh - full) * 1e4:+.2f}   <- cannot be recovered from a real short log (the observations do not exist)")
    print(f"  structure ambiguity (short64 -> structure known): {(sh - ok) * 1e4:+.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    pub = public_dir(a.data); t0 = time.time()
    from method4_ml.model import Model
    _S["gm"] = Model(cache=True)._fit_gate(pub)                      # v8 production gate (baseline cache; fver f3.5)
    _S["data"] = H.load_machines(pub); q, seq = load_split(pub, "train"); _S["seq"] = seq
    ctx = get_context("fork")
    tasks = [(f, m) for f in list(range(NF)) + [None] for m in MODELS]
    with ctx.Pool(8) as pool:
        fits = pool.map(fit_task, tasks, chunksize=1)
    print(f"population fits done in {time.time() - t0:.0f}s (cross-fitted {NF} folds + 1 full fit per model)", flush=True)
    P = {}; full = {}
    for fold, name, x, names, ll, n in fits:
        h = H.Hier(_S["data"], **MODELS[name]); h.names = names
        if fold is None:
            full[name] = dict(zip(names, x), ll=ll, n=n)
        else:
            P[(fold, name)] = (h, np.array(x))
    os.makedirs("experiments", exist_ok=True)
    json.dump(full, open("experiments/hier_fits_full.json", "w"), indent=1)
    _S["params"] = P
    long_rows = [r for r in q if sum(v < 0 for v in seq[r["machine_id"]]) < 0.1 * len(seq[r["machine_id"]])]
    with ctx.Pool(8) as pool:
        out = pool.map(machine_job, [(r,) for r in long_rows], chunksize=4)
    rows = [x for o in out for x in o]
    print(f"{len(rows)} prediction rows in {time.time() - t0:.0f}s total", flush=True)
    report(rows)


if __name__ == "__main__":
    main()
