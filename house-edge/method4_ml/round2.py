"""Round-2 runbook with automatic variant selection. Owner: Person D.

  python -m method4_ml.round2 --data data/adaptation --out build/round2

1. Refits v7 on --data (everything is re-learned from that round's train histories + all histories: structures, cell priors, signal
   shifts, size model, stackers). Nothing is carried over from the main round.
2. Builds four variants from ONE fit (no refit): gate+sizes only / tree stacker / structured logistic correction / v7 (both).
3. If the round's validation targets exist, picks the variant by the TUNING half only (kit folds), reports the audit half once and
   the full validation score for every variant. Reads targets here, outside fit/predict (diagnosis + selection harness).
   No targets -> v7 (the main-round winner). Rule of thumb enforced: a variant must beat 'gate only' on tuning by >= MARGIN to be
   chosen; otherwise the simpler model wins, so a stacker that stops generalizing cannot hurt.
4. Writes <out>/submission.csv (test), <out>/chosen.json (settings for method4_ml.make_entry --kwargs), and a learned-structure
   report with flags, so a changed mechanism is visible at a glance.
"""
import argparse, json, os, time
from common.data import public_dir, write_predictions
from common.evaluate import folds, points
from method4_ml.model import Model

MARGIN = 0.05   # tuning points a variant must gain over 'gate only' to be chosen (selection noise is ~0.03-0.05 on 48 episodes)
VARIANTS = {"gate only": (0.0, 0.0), "trees": (0.4, 0.0), "logistic": (0.0, 1.0), "v7 (trees + logistic)": (0.4, 0.5)}


def report(m):
    gm = m.gm; g = m.size_models["gate"]
    top = sorted(((p, str(s)) for s, p in zip(list(gm.ST) + ["no gate"], gm.sprior)), reverse=True)[:6]
    out = {"win structures (prior, top 6)": [(n, round(p, 3)) for p, n in top],
           "cell priors (type)": {str(k): [round(x, 2) for x in v] for k, v in gm.cellpriors.items() if len(k) == 1},
           "signal shift per type (beta_on, beta_off, evidence gain)": {t: (a, b, round(z, 1)) for t, (a, b, z) in getattr(gm, "signal_shift", {}).items()},
           "p_change_short": getattr(gm, "p_change_short", None),
           "size tie weight": round(g.tie_weight, 3) if isinstance(g.tie_weight, float) else g.tie_weight,
           "size-lag prior [none, lag1..8]": [round(x, 3) for x in g.prior],
           "size levels P(1x|win) lo/hi": {t: [round(x, 3) for x in v] for t, v in (g.levels or {}).items()}}
    flags = []
    if out["win structures (prior, top 6)"][0][1] < .15: flags.append("no dominant win structure: gate family may not match the new mechanism")
    if isinstance(out["size tie weight"], float) and out["size tie weight"] < .5: flags.append("size gate not clearly two-state: free-cell fallback in use")
    if all(b == 0 for _, b, _ in getattr(gm, "signal_shift", {}).values()) and all(a == 0 for a, _, _ in getattr(gm, "signal_shift", {}).values()):
        flags.append("no signal effect learned")
    return out, flags or ["no flags: main-round structure types still hold (values may differ)"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no-cache", action="store_true", help="recompute everything (use for the real run)")
    a = ap.parse_args(); pub = public_dir(a.data); os.makedirs(a.out, exist_ok=False)
    t0 = time.time(); m = Model(cache=not a.no_cache).fit(pub); print(f"fit {time.time() - t0:.0f}s", flush=True)
    rep, flags = report(m); print(json.dumps(rep, indent=1)); print("FLAGS:", *flags, sep="\n  ")
    have = os.path.exists(os.path.join(pub, "validation", "targets.csv"))
    preds = {}
    for name, (b, w) in VARIANTS.items():
        m.blend, m.lr_weight = b, w; preds[name] = m.predict(pub, "validation")
    choice = "v7 (trees + logistic)"
    if have:
        tune, audit = folds(pub); sc = {n: points(pub, p, tune)[0] for n, p in preds.items()}
        print(f"\n{'variant':24s} tuning    audit     full")
        for n, p in preds.items():
            print(f"{n:24s} {sc[n]:.3f}  {points(pub, p, audit)[0]:.3f}  {points(pub, p, None)[0]:.3f}")
        base = sc["gate only"]; best = max(sc, key=sc.get)
        choice = best if sc[best] - base >= MARGIN else "gate only"
        print(f"\nchosen on TUNING half: {choice}  (best {best} {sc[best]:.3f}, gate only {base:.3f}, margin {MARGIN})")
        if getattr(m, "menu", None) is not None:     # diagnostic only: the size-menu decision is made by the no-target self-check in fit
            bb, ww = VARIANTS[choice]; m.blend, m.lr_weight = bb, ww; keep_menu = m.menu; m.menu = None
            p0 = m.predict(pub, "validation"); m.menu = keep_menu
            print(f"size menu self-check: {getattr(m, 'menu_report', {})}")
            print(f"diagnostic, chosen variant WITHOUT size menu: tuning {points(pub, p0, tune)[0]:.3f}  full {points(pub, p0, None)[0]:.3f}")
    b, w = VARIANTS[choice]; m.blend, m.lr_weight = b, w
    write_predictions(os.path.join(a.out, "submission.csv"), m.predict(pub, "test"))
    json.dump({"blend": b, "lr_weight": w, "variant": choice}, open(os.path.join(a.out, "chosen.json"), "w"))
    json.dump({"report": rep, "flags": flags}, open(os.path.join(a.out, "learned.json"), "w"), indent=1)
    print(f"wrote {a.out}/submission.csv  chosen={choice}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
