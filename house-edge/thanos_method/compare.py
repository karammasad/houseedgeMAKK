"""Official-metric comparison of the challenger vs the champion on full validation (diagnosis / promotion evidence only;
nothing here feeds back into the model). Validation has been inspected by the team before: NOT a fresh holdout.
    python -m thanos_method.compare --new egp.pkl --old champion.pkl [--blend .5] [--label ...]
Reports: points (tuning / audit / full), per-condition ratios, paired within-condition episode bootstrap of the points change,
episodes improved, top-5 share, without top 5 / 10, leave-one-out of the top 5, per machine type, error correlation and, if asked,
the ONE pre-registered fixed-weight blend.
"""
import argparse, json, os, pickle
from collections import defaultdict
import numpy as np
from common.evaluate import points, folds, _rows, COLS, CONDITIONS
from common.data import load_split

PUB = "data/main"


def load(p):
    return pickle.load(open(p, "rb"))


def analyse(new, old, label="new vs old", pub=PUB):
    T = _rows(os.path.join(pub, "validation", "targets.csv"))
    C = {r["query_id"]: r["condition"] for r in _rows(os.path.join(pub, "validation", "conditions.csv"))}
    REF = {r["query_id"]: [float(r[c]) for c in COLS] for r in _rows(os.path.join(pub, "validation", "reference.csv"))}
    q, _ = load_split(pub, "validation"); TY = {r["query_id"]: r["machine_type"] for r in q}
    tune, audit = folds(pub)
    se = lambda p, t: float(sum((a - b) ** 2 for a, b in zip(p, t)))
    epq = defaultdict(list); cond = {}
    for t in T:
        tr = [float(t[c]) for c in COLS]; k = t["query_id"]
        epq[t["episode_id"]].append((k, se(old[k], tr), se(new[k], tr), se(REF[k], tr), tr)); cond[t["episode_id"]] = C[k]
    out = {"label": label}
    for nm, pr in (("old", old), ("new", new)):
        out[nm] = dict(tuning=points(pub, pr, tune)[0], audit=points(pub, pr, audit)[0], full=points(pub, pr)[0], ratios=points(pub, pr)[1])
    B = {c: np.mean([np.mean([x[3] for x in epq[e]]) for e in epq if cond[e] == c]) for c in CONDITIONS}
    n = {c: sum(cond[e] == c for e in epq) for c in CONDITIONS}
    R = 60 / out["old"]["full"] - 1; scale = -60 / (1 + R) ** 2
    pts = {e: scale * sum(x[2] - x[1] for x in v) / len(v) / (n[cond[e]] * B[cond[e]] * 4) for e, v in epq.items()}
    eps = sorted(pts, key=lambda e: -pts[e]); arr = np.array([pts[e] for e in eps]); tot = arr.sum()
    rng = np.random.default_rng(0); by = defaultdict(list)
    for e in eps:
        by[cond[e]].append(pts[e])
    bs = np.array([sum(np.mean(rng.choice(v, len(v))) * len(v) for v in by.values()) for _ in range(10000)])
    out["change_full"] = out["new"]["full"] - out["old"]["full"]
    out["bootstrap"] = dict(ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_gain=float((bs > 0).mean()))
    out["episodes_improved"] = f"{int((arr > 0).sum())}/{len(arr)}"
    out["top5_share"] = float(arr[:5].sum() / tot) if tot else None
    out["without_top5"] = float(tot - arr[:5].sum()); out["without_top10"] = float(tot - arr[:10].sum())
    keep = lambda drop: {e for e in epq if e not in drop}
    out["loo_top5"] = [float(points(pub, new, keep({e}))[0] - points(pub, old, keep({e}))[0]) for e in eps[:5]]
    out["by_condition_points"] = {c: float(sum(by[c])) for c in CONDITIONS}
    out["by_condition_ratio_change"] = {c: float(out["new"]["ratios"][c] / out["old"]["ratios"][c] - 1) for c in CONDITIONS}
    ty = defaultdict(float)
    for e, v in epq.items():
        for k, o, nw, _, _ in v:
            ty[TY[k]] += scale * (nw - o) / len(v) / (n[cond[e]] * B[cond[e]] * 4)
    out["by_type_points"] = dict(ty)
    # error correlation (per query, 5-class error vectors vs truth)
    eo = np.array([np.array(old[k]) - np.array(x[4]) for v in epq.values() for x in v for k in [x[0]]])
    en = np.array([np.array(new[k]) - np.array(x[4]) for v in epq.values() for x in v for k in [x[0]]])
    out["error_corr_vector"] = float(np.corrcoef(eo.ravel(), en.ravel())[0, 1])
    out["error_corr_pwin"] = float(np.corrcoef(eo[:, 0], en[:, 0])[0, 1])
    out["sq_err_corr_episode"] = float(np.corrcoef([np.mean([x[1] for x in v]) for v in epq.values()], [np.mean([x[2] for x in v]) for v in epq.values()])[0, 1])
    return out


def show(o):
    print(f"== {o['label']}")
    for nm in ("old", "new"):
        r = o[nm]; print(f"  {nm}: tuning {r['tuning']:.4f}  audit {r['audit']:.4f}  FULL {r['full']:.4f}  ratios " +
                         " ".join(f"{c[:6]} {r['ratios'][c]:.4f}" for c in CONDITIONS))
    b = o["bootstrap"]
    print(f"  change full {o['change_full']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  P(gain>0) {b['p_gain']:.3f}  improved {o['episodes_improved']}")
    print(f"  top-5 share {o['top5_share']:.0%}  without top5 {o['without_top5']:+.4f}  without top10 {o['without_top10']:+.4f}  LOO top5 " +
          " ".join(f"{x:+.4f}" for x in o["loo_top5"]))
    print("  by condition (points): " + " ".join(f"{c} {v:+.4f}" for c, v in o["by_condition_points"].items()))
    print("  by condition (ratio change): " + " ".join(f"{c} {v:+.1%}" for c, v in o["by_condition_ratio_change"].items()))
    print("  by type (points): " + " ".join(f"{c} {v:+.4f}" for c, v in sorted(o["by_type_points"].items())))
    print(f"  error correlation: 5-vector {o['error_corr_vector']:.3f}  p0 (=1-P(win)) {o['error_corr_pwin']:.3f}  episode sq-err {o['sq_err_corr_episode']:.3f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--old", required=True)
    ap.add_argument("--blend", type=float, default=None); ap.add_argument("--label", default=""); ap.add_argument("--json")
    a = ap.parse_args(); new, old = load(a.new), load(a.old)
    res = [analyse(new, old, a.label or "challenger vs champion")]
    show(res[0])
    if a.blend is not None:
        w = a.blend; bl = {k: [w * x + (1 - w) * y for x, y in zip(new[k], old[k])] for k in old}
        res.append(analyse(bl, old, f"blend {w} challenger + {1 - w} champion vs champion")); show(res[-1])
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
