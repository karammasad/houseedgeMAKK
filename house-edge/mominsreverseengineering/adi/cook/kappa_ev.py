"""Train-only evidence for kappa per machine type on short logs (real short train logs + masked long train logs)."""
import json, math, random, sys
from collections import defaultdict
from multiprocessing import get_context
import adi.gate as G
from adi.cook import hier
from adi.model import Model as Adi
import method2_sizes.round2 as R
from common.data import load_split, public_dir

KAPPAS = [None, 3, 10, 30, 100, 300, 1000]
_M = {}


def mask(r, s, seed):
    rng = random.Random(f"{seed}|{r['machine_id']}"); t = len(s)
    keep = set(range(t - 40, t)) | set(rng.sample(range(t - 40), 24))
    return [x if i in keep else -1 for i, x in enumerate(s)]


def job(args):
    r, s = args; m = _M["m"]; out = []
    for k in KAPPAS:
        hier.install(k, None)
        out.append(G.win_probability(r, s, m.ST, m._sp(s), cellprior=m._cp(r), evidence=True,
                                     **m._mcfg(s, r_type=r["machine_type"]))[1])
    hier.install(None)
    return r["machine_type"], out


if __name__ == "__main__":
    pub = public_dir("data/main"); m = Adi(**R.ADI).fit(pub); _M["m"] = m
    q, seq = load_split(pub, "train"); jobs = []
    for r in q:
        s = seq[r["machine_id"]]
        if min(s) < 0:
            jobs.append((r, s))
        else:
            for sd in range(2):
                jobs.append((r, mask(r, s, sd)))
    with get_context("fork").Pool(12) as p:
        res = p.map(job, jobs, chunksize=8)
    tot = defaultdict(lambda: [0.0] * len(KAPPAS)); n = defaultdict(int)
    for t, v in res:
        n[t] += 1; tot[t] = [a + b for a, b in zip(tot[t], v)]
    for t in sorted(tot):
        print(t, n[t], " ".join(f"{k}:{x - tot[t][0]:+.2f}" for k, x in zip(KAPPAS, tot[t])))
