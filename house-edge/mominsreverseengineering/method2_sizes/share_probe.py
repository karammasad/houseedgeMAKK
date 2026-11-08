"""Short-history probe: is the size-gate lag / direction shared with anything a short machine can borrow from?
Train histories only (long-history machines, where the gate is identifiable).
  python3 -m method2_sizes.share_probe --data data/main
For each machine: posterior over size-gate hypotheses (GateSizes, npmle cells); confident machines only.
Direction = whether a 1x at the gate lag puts the machine in the high-1x state.
"""
import argparse, csv, os, math, itertools
from collections import defaultdict, Counter
from common.data import load_split, public_dir
from method2_sizes.sizes import GateSizes
from method1_gate.gate import structures, machine_tables


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); q, seq = load_split(pub, "train")
    with open(os.path.join(pub, "train", "conditions.csv")) as f:
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}
    g = GateSizes(cell_prior="npmle").fit(q, seq)
    ST = structures(8, 2)
    info = {}
    for r in q:
        if cond[r["query_id"]] == "short_history":
            continue
        s = seq[r["machine_id"]]; t = r["machine_type"]
        tabs = g._tables(s, g.mu[t], t); lw = [math.log(p) + ll for p, (ll, _) in zip(g.prior, tabs)]
        m = max(lw); w = [math.exp(x - m) for x in lw]; z = sum(w); post = [x / z for x in w]
        k = max(range(len(post)), key=post.__getitem__)
        if post[k] < 0.9:
            continue
        direction = None
        if k > 0:
            cells = g._cells(s)[k][0]
            a1 = (cells[1][1] + 1) / (sum(cells[1]) + 2); a2 = (cells[2][1] + 1) / (sum(cells[2]) + 2)
            direction = int(a1 > a2)
        wt = machine_tables(s, len(s), [], ST, .5, .5); wl = [x[0][0] for x in wt]
        info[r["machine_id"]] = dict(ep=r["episode_id"], typ=t, sig=r["signal"], lag=k, dir=direction,
                                     wgate=(ST + [()])[max(range(len(wl)), key=wl.__getitem__)])
    print(f"confident machines: {len(info)} (posterior >= 0.9)")
    print("size-gate lag (0 = none) distribution:", Counter(v["lag"] for v in info.values()).most_common())
    by_ep = defaultdict(list)
    for v in info.values():
        by_ep[v["ep"]].append(v)
    same = tot = 0; dsame = dtot = 0
    for vs in by_ep.values():
        for x, y in itertools.combinations(vs, 2):
            tot += 1; same += x["lag"] == y["lag"]
            if x["lag"] == y["lag"] and x["lag"] > 0:
                dtot += 1; dsame += x["dir"] == y["dir"]
    pl = Counter(v["lag"] for v in info.values()); n = len(info)
    chance = sum((c / n) ** 2 for c in pl.values())
    print(f"same lag within episode: {same / tot:.3f}  (chance {chance:.3f})")
    print(f"same direction given same lag within episode: {dsame / max(1, dtot):.3f}  (chance ~0.5, n={dtot})")
    for key in ("typ", "sig"):
        tab = defaultdict(Counter)
        for v in info.values():
            tab[v[key]][v["lag"]] += 1
        print(f"lag by {key}:", {k: dict(c.most_common(4)) for k, c in sorted(tab.items())})
        dt = defaultdict(Counter)
        for v in info.values():
            if v["dir"] is not None:
                dt[v[key]][v["dir"]] += 1
        print(f"direction by {key}:", {k: dict(c) for k, c in sorted(dt.items())})
    rel = Counter()
    for v in info.values():
        if v["lag"] > 0 and v["wgate"]:
            rel["size lag in win-gate lags" if v["lag"] in v["wgate"] else "not in win-gate lags"] += 1
    print("size lag vs win gate:", dict(rel), " win gates:", Counter(v["wgate"] for v in info.values()).most_common(4))
    j = Counter((v["wgate"], v["lag"]) for v in info.values() if v["lag"] > 0)
    print("joint (win gate, size lag) top:", j.most_common(9))


if __name__ == "__main__":
    main()
