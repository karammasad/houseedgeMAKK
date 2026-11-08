"""Train-only evidence for kappa per machine type on LONG logs."""
from collections import defaultdict
from multiprocessing import get_context
import adi.cook.kappa_ev as E
from adi.model import Model as Adi
import method2_sizes.round2 as R
from common.data import load_split, public_dir
import adi.gate as G
from adi.cook import hier


def job(args):
    r, s = args; m = E._M["m"]; out = []
    for k in E.KAPPAS:
        hier.install(None, k)
        out.append(G.win_probability(r, s, m.ST, m._sp(s), cellprior=m._cp(r), evidence=True,
                                     **m._mcfg(s, r_type=r["machine_type"]))[1])
    hier.install(None)
    return r["machine_type"], out


if __name__ == "__main__":
    pub = public_dir("data/main"); E._M["m"] = Adi(**R.ADI).fit(pub)
    q, seq = load_split(pub, "train")
    jobs = [(r, seq[r["machine_id"]]) for r in q if min(seq[r["machine_id"]]) >= 0]
    with get_context("fork").Pool(10) as p:
        res = p.map(job, jobs, chunksize=4)
    tot = defaultdict(lambda: [0.0] * len(E.KAPPAS))
    for t, v in res:
        tot[t] = [a + b for a, b in zip(tot[t], v)]
    for t in sorted(tot):
        print(t, " ".join(f"{k}:{x - tot[t][0]:+.2f}" for k, x in zip(E.KAPPAS, tot[t])))
