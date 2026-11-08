"""Sign-tie experiment (FINDINGS.md Entry 4). TRAINING HISTORIES ONLY: no validation file, no targets, no audit.

Claim under test: ONE hidden binary state, identified from PAYOUT SIZES alone, shifts the WIN logit with a type-independent sign
(the state with bigger payouts has the higher win rate).

Per stable train machine (condition labels pick the subset only; never an input to anything):
  first half  -> (a) gate pair by max-likelihood parity scan over all pairs to lag 12, on-cell = higher-rate cell;
                 (b) size lag k and category grouping g by the split of P(1x | win) in the first half, "big-size state" = the side with
                     the LOWER 1x share. No win-rate-by-state information is used for (b).
  second half -> in on-cell ticks, D = logit(win rate | big-size state) - logit(win rate | other state), add-half smoothing.
Controls: N1 decoy lags (not i, j, k); N2 state rule borrowed from another machine of the same type with a DIFFERENT size lag;
extra: N3 time-shifted state (label from t-13/17/23), N4 labels permuted among the machine's own on-cell ticks (estimator null).
Selection into the analysis uses first-half sizes only, so it cannot bias second-half win rates.
Run:  python3 signtie.py --data ../data/main
"""
import argparse, csv, math, os, random, zlib
from collections import defaultdict
from common.data import public_dir, load_split

MAXLAG, GATE_LAGS = 8, 12
# DEVIATION from the first draft (8.0 -> 5.0): the first run identified 2 of 192 machines, before any D was computed. Half-history
# size evidence is only ~45 wins, so 8 nats is unreachable for steady machines. Selection uses first-half sizes only, so the change
# cannot bias the second-half win rates; false identifications only dilute D toward 0 (conservative).
MIN_GATE_GAIN, MIN_SIZE_GAIN, MIN_SIDE_N, MIN_CELL_N = 15.0, 5.0, 8, 6


def ll(k, n):
    if n == 0:
        return 0.0
    r = k / n
    return (k * math.log(r) if k else 0.0) + ((n - k) * math.log(1 - r) if n - k else 0.0)


def gain(a, b):
    return ll(*a) + ll(*b) - ll(a[0] + b[0], a[1] + b[1])


def cat(v):
    return 0 if v == 0 else (1 if v == 1 else 2)


def logit(p):
    return math.log(p / (1 - p))


def identify(seq, half, tie_k=False):
    """From the FIRST half only. Returns None or dict(pair, on_value, k, g, low_side, gates, size_gain)."""
    w = [-1 if v < 0 else int(v > 0) for v in seq]
    best, second = (None, 0.0), 0.0
    for i in range(1, GATE_LAGS + 1):
        for j in range(i + 1, GATE_LAGS + 1):
            c = [[0, 0], [0, 0]]
            for t in range(GATE_LAGS, half):
                if w[t] < 0 or w[t - i] < 0 or w[t - j] < 0:
                    continue
                p = w[t - i] ^ w[t - j]; c[p][0] += w[t]; c[p][1] += 1
            g = gain(tuple(c[0]), tuple(c[1]))
            if g > best[1]:
                second = best[1]; best = ((i, j, c), g)
            elif g > second:
                second = g
    if best[0] is None or best[1] < MIN_GATE_GAIN:
        return None
    i, j, c = best[0]
    on_value = 0 if (c[0][0] / max(c[0][1], 1)) > (c[1][0] / max(c[1][1], 1)) else 1
    stats = defaultdict(lambda: [[0, 0], [0, 0]])          # (k, g) -> side -> [n 1x, n wins]
    for t in range(MAXLAG, half):
        if seq[t] <= 0:
            continue
        for k in ((min(best[0][0], best[0][1]),) if tie_k else range(1, MAXLAG + 1)):
            if seq[t - k] < 0:
                continue
            for g in range(3):
                side = int(cat(seq[t - k]) == g); s = stats[(k, g)][side]
                s[0] += seq[t] == 1; s[1] += 1
    cand = []
    for (k, g), sd in stats.items():
        if min(sd[0][1], sd[1][1]) >= MIN_SIDE_N:
            cand.append((gain(tuple(sd[0]), tuple(sd[1])), k, g, sd))
    if not cand:
        return None
    gsz, k, g, sd = max(cand, key=lambda x: x[0])
    if gsz < MIN_SIZE_GAIN:
        return None
    low_side = 0 if sd[0][0] / sd[0][1] < sd[1][0] / sd[1][1] else 1     # side with the lower 1x share = big-size state
    return dict(pair=(i, j), on_value=on_value, k=k, g=g, low_side=low_side, gate_gain=best[1], size_gain=gsz)


def state_fn(seq, k, g, low_side, shift=0):
    def f(t):
        s = t - shift - k
        if s < 0 or seq[s] < 0:
            return None
        return int(int(cat(seq[s]) == g) == low_side)
    return f


def second_half(seq, half, info, fn, cells=("on", "off"), shuffle_rng=None):
    """counts[cell][h] = [wins, ticks] over the second half, state from fn(t)."""
    i, j = info["pair"]; w = [-1 if v < 0 else int(v > 0) for v in seq]
    rows = []
    for t in range(half, len(seq)):
        if w[t] < 0 or w[t - i] < 0 or w[t - j] < 0:
            continue
        h = fn(t)
        if h is None:
            continue
        cell = "on" if (w[t - i] ^ w[t - j]) == info["on_value"] else "off"
        if cell in cells:
            rows.append([cell, h, w[t]])
    if shuffle_rng is not None:                              # N4: permute state labels among each cell's own ticks
        for cell in cells:
            idx = [r for r in rows if r[0] == cell]; hs = [r[1] for r in idx]; shuffle_rng.shuffle(hs)
            for r, h in zip(idx, hs):
                r[1] = h
    out = {c: {0: [0, 0], 1: [0, 0]} for c in cells}
    for cell, h, y in rows:
        out[cell][h][0] += y; out[cell][h][1] += 1
    return out


def dval(cnt):
    (k0, n0), (k1, n1) = cnt[0], cnt[1]
    if min(n0, n1) < MIN_CELL_N:
        return None
    return logit((k1 + .5) / (n1 + 1)) - logit((k0 + .5) / (n0 + 1))


def boot_ci(vals, eps, reps=2000, seed=7):
    """Cluster bootstrap over episodes of the mean of per-machine values."""
    by = defaultdict(list)
    for v, e in zip(vals, eps):
        by[e].append(v)
    keys = sorted(by); rng = random.Random(seed); ms = []
    for _ in range(reps):
        draw = [v for e in (rng.choice(keys) for _ in keys) for v in by[e]]
        ms.append(sum(draw) / len(draw))
    ms.sort()
    return ms[int(.025 * reps)], ms[int(.975 * reps)]


def summarize(name, rec):
    out = []
    for ty in ("ALL", "balanced", "jackpot", "steady"):
        sel = [(v, e) for (t, e, v) in rec if v is not None and (ty == "ALL" or t == ty)]
        if len(sel) < 5:
            out.append(f"  {name:34s} {ty:9s} n={len(sel):3d}  (too few)"); continue
        v = [x for x, _ in sel]; e = [y for _, y in sel]; m = sum(v) / len(v)
        lo, hi = boot_ci(v, e)
        pos = sum(x > 0 for x in v) / len(v)
        out.append(f"  {name:34s} {ty:9s} n={len(v):3d}  mean D {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  share D>0 {pos:.0%}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
    ap.add_argument("--tie-k", action="store_true", help="EXPLORATORY follow-up: size lag fixed to the smaller gate lag (documented tie)")
    a = ap.parse_args()
    pub = public_dir(a.data)
    q, seq = load_split(pub, "train")
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "train", "conditions.csv")))}
    rows = [r for r in q if cond.get(r["query_id"]) == "stable"]       # subset selection only
    print(f"train stable machines: {len(rows)} (history only; no labels, no validation, no audit)")
    ident = {}
    for r in rows:
        s = seq[r["machine_id"]]
        if len(s) < 100 or sum(1 for v in s if v < 0) > 0.1 * len(s):
            continue
        info = identify(s, len(s) // 2, a.tie_k)
        if info:
            ident[r["machine_id"]] = (r, info)
    print(f"identified from FIRST half (gate gain >= {MIN_GATE_GAIN}, size-split gain >= {MIN_SIZE_GAIN}): {len(ident)} machines")
    agree = sum(1 for _, (r, f) in ident.items() if f["k"] == min(f["pair"]))
    print(f"sanity: size lag k equals the smaller gate lag for {agree}/{len(ident)} machines (prior finding: tied)")
    menu = defaultdict(int)
    for _, (r, f) in ident.items():
        menu[f["pair"]] += 1
    print("pairs found (count):", dict(sorted(menu.items(), key=lambda kv: -kv[1])[:8]))

    rec = {n: [] for n in ("main_on", "main_off", "N1_decoy_on", "N2_other_machine_on", "N3_timeshift_on", "N4_permuted_on")}
    by_type = defaultdict(list)
    for m, (r, f) in sorted(ident.items()):
        by_type[r["machine_type"]].append(m)
    for m, (r, f) in sorted(ident.items()):
        s = seq[m]; half = len(s) // 2; ty, ep = r["machine_type"], r["episode_id"]
        main = second_half(s, half, f, state_fn(s, f["k"], f["g"], f["low_side"]))
        rec["main_on"].append((ty, ep, dval(main["on"]))); rec["main_off"].append((ty, ep, dval(main["off"])))
        ds = []
        for kk in range(1, MAXLAG + 1):                       # N1: decoy lags
            if kk in (f["k"], *f["pair"]):
                continue
            d = dval(second_half(s, half, f, state_fn(s, kk, f["g"], f["low_side"]), cells=("on",))["on"])
            if d is not None:
                ds.append(d)
        rec["N1_decoy_on"].append((ty, ep, sum(ds) / len(ds) if ds else None))
        rng = random.Random(zlib.crc32((m + '|1').encode()))            # N2: another machine's rule, different size lag
        cand = [o for o in by_type[ty] if o != m and ident[o][1]["k"] != f["k"]]
        d2 = None
        if cand:
            of = ident[rng.choice(sorted(cand))][1]
            d2 = dval(second_half(s, half, f, state_fn(s, of["k"], of["g"], of["low_side"]), cells=("on",))["on"])
        rec["N2_other_machine_on"].append((ty, ep, d2))
        ds = [dval(second_half(s, half, f, state_fn(s, f["k"], f["g"], f["low_side"], sh), cells=("on",))["on"]) for sh in (13, 17, 23)]
        ds = [d for d in ds if d is not None]
        rec["N3_timeshift_on"].append((ty, ep, sum(ds) / len(ds) if ds else None))
        rec["N4_permuted_on"].append((ty, ep, dval(second_half(s, half, f, state_fn(s, f["k"], f["g"], f["low_side"]), cells=("on",),
                                                                 shuffle_rng=random.Random(zlib.crc32((m + '|4').encode())))["on"])))
    print("\nD = logit win rate(big-size state) - logit win rate(other state), second half, per machine; prediction: D ~ +0.5..+0.9 on-cells")
    for n in rec:
        print(*summarize(n, rec[n]), sep="\n"); print()
    # pre-registered verdict (FINDINGS Entry 4): falsified if main D inside +-0.15 or a control matches it
    v = [x for (_, _, x) in rec["main_on"] if x is not None]; e = [y for (_, y, x) in rec["main_on"] if x is not None]
    if len(v) >= 10:
        m = sum(v) / len(v); lo, hi = boot_ci(v, e)
        ctrl = {}
        for n in ("N1_decoy_on", "N2_other_machine_on"):
            cv = [x for (_, _, x) in rec[n] if x is not None]; ctrl[n] = sum(cv) / len(cv) if cv else float("nan")
        passed = lo > 0.15 and all(abs(c) <= 0.15 for c in ctrl.values())
        print(f"VERDICT (pre-registered): main mean D {m:+.3f} CI [{lo:+.3f}, {hi:+.3f}]; controls " +
              ", ".join(f"{k} {c:+.3f}" for k, c in ctrl.items()) + f" -> {'PASS' if passed else 'NOT PASSED (read the table)'}")
    else:
        print("VERDICT: too few identified machines")


if __name__ == "__main__":
    main()
