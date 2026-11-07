"""Posterior predictive check: simulate histories from the fitted generative model and compare their observable
statistics with REAL held-out histories.

  python -m method2_sizes.ppc PARAMS.json --data data/main [--reps 2] [--shift jackpot=0.4,balanced=0.2]

Real histories = validation + test (episodes disjoint from train, where the parameters were fitted); only histories
are read, never labels/targets/conditions. Each real machine is a template for (type, signal, cutoff, observation
mask), so short logs keep their 40 + 24 layout. Z = (real - sim) / sqrt(se_real^2 + se_sim^2), SEs by bootstrap
over machines. |Z| > 3 is flagged as a misspecified feature.
"""
import argparse, json, math, random
from collections import defaultdict
from common.data import load_split, public_dir
from method2_sizes.genmodel import Simulator, machine_stats, is_short

STATS = [("rate", "mean", "win rate"), ("rate", "sd", "win rate spread (sd)"),
         ("best_e", "q10", "best-pair effect q10"), ("best_e", "q50", "best-pair effect median"),
         ("best_e", "q90", "best-pair effect q90"),
         ("hi_rate", "q10", "high-cell rate q10"), ("hi_rate", "q50", "high-cell rate median"),
         ("hi_rate", "q90", "high-cell rate q90"), ("lo_rate", "q10", "low-cell rate q10"),
         ("lo_rate", "q50", "low-cell rate median"), ("lo_rate", "q90", "low-cell rate q90"),
         ("single_e", "mean", "mean |single-lag effect|"), ("flip", "mean", "best-pair sign flips between halves"),
         ("size_e", "q50", "size-lag effect median"), ("size_e", "q90", "size-lag effect q90"),
         ("link", "mean", "win rate | big@size lag (sign-adj.)"),
         ("share_1", "mean", "share 1x among wins"), ("share_20", "mean", "share 20x among wins"),
         ("win_run", "mean", "mean win run"), ("loss_run", "mean", "mean loss run")]


def summ(v, how):
    v = [x for x in v if x is not None]
    if len(v) < 5:
        return None
    if how == "mean":
        return sum(v) / len(v)
    if how == "sd":
        m = sum(v) / len(v); return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    q = {"q10": .1, "q50": .5, "q90": .9}[how]; s = sorted(v); return s[min(len(s) - 1, int(q * len(s)))]


def boot_se(v, how, rng, B=60):
    v = [x for x in v if x is not None]
    if len(v) < 5:
        return None
    vals = [summ([rng.choice(v) for _ in v], how) for _ in range(B)]
    m = sum(vals) / B; return math.sqrt(sum((x - m) ** 2 for x in vals) / (B - 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("params"); ap.add_argument("--data", default="data/main")
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--shift", default="")
    ap.add_argument("--p_flip", type=float, default=.4); ap.add_argument("--scheme", default="legacy"); ap.add_argument("--only", default="")
    a = ap.parse_args(); pub = public_dir(a.data); par = json.load(open(a.params))
    shift = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.shift.split(",") if kv}
    real = []
    for split in ("validation", "test"):
        q, seq = load_split(pub, split)
        real += [(r, seq[r["machine_id"]]) for r in q]
    print(f"{len(real)} real held-out machines; simulating {a.reps} replicate(s) each; p_flip={a.p_flip}, shift={shift}")
    sim = Simulator(par, p_flip=a.p_flip, state_win_shift=shift, seed=7); rng = random.Random(11)
    grp = lambda r, s: (r["machine_type"], "short" if is_short(s) else "long")
    R, S = defaultdict(list), defaultdict(list)
    for r, s in real:
        R[grp(r, s)].append(machine_stats(s))
    # shared episodes: machines of an episode share one flip time
    eps = defaultdict(list)
    for r, s in real:
        eps[r["episode_id"]].append((r, s))
    for rep in range(a.reps):
        for ep, items in eps.items():
            short_ep = is_short(items[0][1]); cut = items[0][0]["cutoff"]
            if a.scheme == "learned":   # measured from train: stable never flips; drift & shared always, at ~0.69 cutoff
                kind = "short" if short_ep else sim.rng.choice(("stable", "drift", "shared"))
                shared_tau = round(sim.rng.gauss(.685, .045) * cut) if kind == "shared" else None
            for r, s in items:
                mask = [v >= 0 for v in s]
                if a.scheme == "learned":
                    if kind == "stable":
                        tau = None
                    elif kind == "drift":
                        tau = round(min(.85, max(.5, sim.rng.gauss(.70, .06))) * cut)
                    elif kind == "shared":
                        tau = shared_tau
                    else:
                        tau = round(sim.rng.uniform(.3, .9) * cut) if sim.rng.random() < .05 else None
                    m = sim.machine(r["machine_type"], r["signal"], cut, mask, tau=tau)
                else:
                    sh = sim.rng.randint(40, max(41, cut - 10)) if sim.rng.random() < sim.shared_frac else None
                    m = sim.machine(r["machine_type"], r["signal"], cut, mask, sh)
                S[grp(r, s)].append(machine_stats(m))
    flagged = []
    for g in sorted(R):
        if a.only and a.only not in g[0] + g[1]:
            continue
        print(f"\n=== {g[0]} / {g[1]}: real n={len(R[g])}, sim n={len(S[g])} ===")
        print(f"  {'statistic':44s} {'real':>8s} {'sim':>8s} {'Z':>6s}")
        for key, how, label in STATS:
            rv, sv = [m[key] for m in R[g]], [m[key] for m in S[g]]
            rs, ss = summ(rv, how), summ(sv, how)
            if rs is None or ss is None:
                continue
            se = math.sqrt((boot_se(rv, how, rng) or 0) ** 2 + (boot_se(sv, how, rng) or 0) ** 2) or 1e-9
            z = (rs - ss) / se; flag = " <-- FAIL" if abs(z) > 3 else ""
            print(f"  {label:44s} {rs:8.3f} {ss:8.3f} {z:+6.1f}{flag}")
            if abs(z) > 3:
                flagged.append((abs(z), g, label, rs, ss))
    print(f"\n{len(flagged)} flagged statistics (|Z| > 3). Largest:")
    for z, g, label, rs, ss in sorted(flagged, reverse=True)[:12]:
        print(f"  |Z|={z:5.1f}  {g[0]}/{g[1]}  {label}: real {rs:.3f} vs sim {ss:.3f}")


if __name__ == "__main__":
    main()
