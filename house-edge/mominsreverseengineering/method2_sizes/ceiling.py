"""Self-consistency ceiling: how well does our inference do in a world that IS the generative model?

  python -m method2_sizes.ceiling MODEL.pkl PARAMS.json --data data/main [--reps 1]

Simulates held-out-like machines (templates = real validation machines: type, signal, cutoff, observation mask;
flip scheme and size-win link as validated by the PPC), keeps the TRUE next-spin distribution, runs the fitted
FINAL model (adi gate + method 2 sizes) on the simulated histories, and compares its mean squared probability
error with its error on the REAL validation set. If sim error ~ real error, the model is at the ceiling its own
assumptions allow; a large gap would mean real structure we do not model.
"""
import argparse, csv, json, os, pickle, random, sys
from collections import defaultdict
from common.data import load_split, public_dir
from common.evaluate import COLS
from method2_sizes.genmodel import Simulator, is_short
from adi import gate as G

SHIFT = {"jackpot": -0.43, "balanced": -0.27, "steady": -0.15}


def mse(p, t):
    s = sum(p); return sum((x / s - y) ** 2 for x, y in zip(p, t))


def predict(win, sizes, r, s):
    pw = G.win_probability(r, s, win.ST, win._sp(s), cellprior=win._cp(r), **win._mcfg(s, r_type=r["machine_type"]))
    return [1 - pw] + [pw * x for x in sizes.predict(r, s)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkl"); ap.add_argument("params"); ap.add_argument("--data", default="data/main")
    ap.add_argument("--reps", type=int, default=1); a = ap.parse_args(); sys.setrecursionlimit(10000)
    pub = public_dir(a.data); m = pickle.load(open(a.pkl, "rb")); par = json.load(open(a.params))
    vq, vs = load_split(pub, "validation")
    rd = lambda f: list(csv.DictReader(open(os.path.join(pub, "validation", f))))
    T = {r["query_id"]: [float(r[c]) for c in COLS] for r in rd("targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in rd("conditions.csv")}
    sim = Simulator(par, state_win_shift=SHIFT, seed=21)
    eps = defaultdict(list)
    for r in vq:
        eps[r["episode_id"]].append(r)
    real, simd = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in vq:   # real errors of the same model, per condition and episode
        real[cond[r["query_id"]]][r["episode_id"]].append(mse(predict(m.win, m.sizes, r, vs[r["machine_id"]]), T[r["query_id"]]))
    for rep in range(a.reps):
        for ep, rs in eps.items():
            cut = rs[0]["cutoff"]; short = is_short(vs[rs[0]["machine_id"]])
            kind = "short_history" if short else sim.rng.choice(("stable", "drift", "shared_change"))
            shared_tau = round(sim.rng.gauss(.685, .045) * cut) if kind == "shared_change" else None
            for r in rs:
                if kind == "stable":
                    tau = None
                elif kind == "drift":
                    tau = round(min(.85, max(.5, sim.rng.gauss(.70, .06))) * cut)
                elif kind == "shared_change":
                    tau = shared_tau
                else:
                    tau = round(sim.rng.uniform(.3, .9) * cut) if sim.rng.random() < .05 else None
                mask = [v >= 0 for v in vs[r["machine_id"]]]
                seq, truth = sim.machine(r["machine_type"], r["signal"], cut, mask, tau=tau, with_truth=True)
                simd[kind][(ep, rep)].append(mse(predict(m.win, m.sizes, r, seq), truth))
    rng = random.Random(5)
    def se(v, B=400):
        ms = [sum(rng.choice(v) for _ in v) / len(v) for _ in range(B)]; mu = sum(ms) / B
        return (sum((x - mu) ** 2 for x in ms) / (B - 1)) ** .5
    print(f"{'condition':14s} {'REAL err (se)':>18s} {'SIM err (se)':>18s} {'sim/real':>9s} {'z':>6s}   (mean over episodes of mean squared prob. error)")
    tot_r = tot_s = 0
    for c in ("stable", "short_history", "drift", "shared_change"):
        er = [sum(v) / len(v) for v in real[c].values()]; es = [sum(v) / len(v) for v in simd[c].values()]
        mr, ms = sum(er) / len(er), sum(es) / len(es); tot_r += mr / 4; tot_s += ms / 4
        sr, ss = se(er), se(es); z = (mr - ms) / (sr ** 2 + ss ** 2) ** .5
        print(f"{c:14s} {mr:10.5f} ({sr:.5f}) {ms:10.5f} ({ss:.5f}) {ms / mr:9.2f} {z:+6.1f}   (episodes: real {len(er)}, sim {len(es)})")
    print(f"{'mean':14s} {tot_r:10.5f} {tot_s:10.5f} {tot_s / tot_r:9.2f}")


if __name__ == "__main__":
    main()
