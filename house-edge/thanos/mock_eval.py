"""Score thanos variants on the mock round-2 kits (official metric, 40-point scale, full validation + per condition).

  python3 -m thanos.mock_eval --kits build/mock_r2 --runs build/mock_r2_runs
Expects <runs>/<scenario>/<variant>/validation.pkl written by thanos.run --also validation.
Also prints the screen support each variant learned, and a paired episode bootstrap of new vs Round 1.
"""
import argparse, json, os, pickle, random
from collections import defaultdict
from common.evaluate import points, CONDITIONS, _rows, COLS


def episode_err(pub, pred):
    T = _rows(os.path.join(pub, "validation", "targets.csv"))
    cond = {r["query_id"]: r["condition"] for r in _rows(os.path.join(pub, "validation", "conditions.csv"))}
    e = defaultdict(list); ce = {}
    for t in T:
        p = pred[t["query_id"]]; s = sum(p)
        e[t["episode_id"]].append(sum((a / s - float(t[c])) ** 2 for a, c in zip(p, COLS))); ce[t["episode_id"]] = cond[t["query_id"]]
    return {k: sum(v) / len(v) for k, v in e.items()}, ce


def boot(pub, new, old, n=2000, seed=7):
    """Paired episode bootstrap within condition of the mean-over-conditions error difference (new - old); < 0 = new better."""
    a, ce = episode_err(pub, new); b, _ = episode_err(pub, old); by = defaultdict(list)
    for k in a:
        by[ce[k]].append(a[k] - b[k])
    rng = random.Random(seed); stat = lambda d: sum(sum(v) / len(v) for v in d.values()) / len(d)
    obs = stat(by); bs = sorted(stat({c: [rng.choice(v) for _ in v] for c, v in by.items()}) for _ in range(n))
    return obs, bs[int(.025 * n)], bs[int(.975 * n)], sum(x < 0 for x in bs) / n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kits", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--base", default="r1"); a = ap.parse_args(); out = {}
    for sc in sorted(os.listdir(a.runs)):
        pub = os.path.join(a.kits, sc, "public")
        if not os.path.isdir(pub):
            continue
        print(f"\n=== {sc}")
        preds = {}
        for v in sorted(os.listdir(os.path.join(a.runs, sc))):
            f = os.path.join(a.runs, sc, v, "validation.pkl")
            if os.path.exists(f):
                preds[v] = pickle.load(open(f, "rb"))
        for v, P in preds.items():
            pts, r = points(pub, P, None, 40.0)
            rj = json.load(open(os.path.join(a.runs, sc, v, "run.json")))
            print(f"  {v:8s} {pts:7.3f}/40  " + "  ".join(f"{c[:6]} {r[c]:.3f}" for c in CONDITIONS) + f"   wall {rj['wall_s']:.0f}s")
            out.setdefault(sc, {})[v] = dict(points=pts, ratios=r, wall=rj["wall_s"])
        for v in preds:
            if v != a.base and a.base in preds:
                d, lo, hi, p = boot(pub, preds[v], preds[a.base])
                print(f"  {v} - {a.base}: error {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  P(better) {p:.3f}")
                out[sc][v]["vs_base"] = dict(diff=d, lo=lo, hi=hi, p=p)
    json.dump(out, open(os.path.join(a.runs, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
