"""Round-two runbook in one command: refit, rediscover, and diff what changed.

  python -m method2_sizes.round2 --data data/adaptation --baseline data/main --out build/round2

(a) Refits the recommended model (FINAL below) on --data, writes <out>/submission.csv (test), and scores
    validation if targets.csv is present.
(b) Reruns the structure-discovery scripts on --data train histories, saving output to <out>/discovery_*.txt.
(c) Prints a diff of everything the model learned, baseline round vs new round, with trust flags:
      - tie weight < 0.5      -> sizes are not clearly two-state; free-cell fallback is doing the work
      - coupling table flat   -> size lag no longer follows the win gate; coupling effectively off
      - big-split concentration low -> big-win split varies by machine; per-machine counts in use
Everything is re-learned from the round's own train histories; nothing is carried over from main.
"""
import argparse, contextlib, io, json, os, subprocess, sys, time
from common.data import public_dir, write_predictions
from common.evaluate import folds, points, CONDITIONS

ADI = {"max_lag": 12, "and_pairs": True, "prior_flips": True, "prune": 0.01, "floor": 0.05,
       "cell_prior": "learn", "cell_group": "type", "p_change_short": "learn"}
FINAL = dict(sizes="gate", gate_cell_prior="npmle", gate_couple=True, gate_couple_iters=20, gate_tied=True,
             gate_tie_weight="learn", gate_joint_em=True, win_model="adi", win_kwargs=ADI)


def learned(model):
    g = model.sizes; w = model.win
    st = getattr(w, "ST", []); sp = getattr(w, "sprior", [])
    top_win = sorted(((p, str(s)) for s, p in zip(list(st) + ["none"], sp)), reverse=True)[:6]
    cond_top = []
    for i, s in enumerate(list(g.wST) + [()]):
        if g.wprior[i] > .03:
            row = g.cond[i]; k = max(range(len(row)), key=row.__getitem__)
            cond_top.append((str(s), "none" if k == 0 else f"lag {k}", round(row[k], 2)))
    return {
        "win structures (adi prior, top 6)": [(n, round(p, 3)) for p, n in top_win],
        "win cell priors (adi)": {str(k): [round(x, 2) for x in v] for k, v in (getattr(w, "cellpriors", None) or {}).items()},
        "p_change_short (adi)": getattr(w, "p_change_short", None),
        "size-lag prior [none, lag1..8]": [round(x, 3) for x in g.prior],
        "size levels P(1x|win) lo/hi": {t: [round(x, 3) for x in v] for t, v in (g.levels or {}).items()},
        "size partition prior [big|1x|loss]": [round(x, 3) for x in (g.part_prior or [])],
        "size tie weight (two-state)": round(g.tie_weight, 3) if isinstance(g.tie_weight, float) else g.tie_weight,
        "coupling win structure -> size lag (P)": cond_top,
        "big-win split 2x/5x/20x": {t: [round(x, 3) for x in v] for t, v in g.split.items()},
        "big-split concentration": {t: round(v, 1) for t, v in g.split_conc.items()},
    }


def flags(L):
    out = []
    tw = L["size tie weight (two-state)"]
    if isinstance(tw, float) and tw < .5:
        out.append(f"tie weight {tw}: sizes not clearly two-state -> free-cell fallback active (OK, automatic)")
    if L["coupling win structure -> size lag (P)"] and max(p for _, _, p in L["coupling win structure -> size lag (P)"]) < .5:
        out.append("coupling table flat: size lag does not follow the win gate -> coupling effectively off")
    for t, c in L["big-split concentration"].items():
        if c < 20:
            out.append(f"{t}: big-win split varies by machine (concentration {c}) -> per-machine counts in use")
    return out or ["no flags: main-round structure types still hold (values may differ; see diff)"]


def fit(pub):
    from method2_sizes.model import Model
    t0 = time.time(); m = Model(**FINAL).fit(pub); return m, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--baseline", default="data/main")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); pub = public_dir(a.data); os.makedirs(a.out, exist_ok=True)

    print(f"== (a) refit on {pub}")
    m, secs = fit(pub)
    sub = os.path.join(a.out, "submission.csv"); write_predictions(sub, m.predict(pub, "test"))
    print(f"  fit {secs:.0f}s; wrote {sub}")
    if os.path.exists(os.path.join(pub, "validation", "targets.csv")):
        pred = m.predict(pub, "validation"); tune, audit = folds(pub)
        maxp = json.load(open(os.path.join(pub, "metadata.json"))).get("max_points", 60)
        fp, fr = points(pub, pred, None, maxp)
        print(f"  validation: tuning {points(pub, pred, tune, maxp)[0]:.3f}  audit {points(pub, pred, audit, maxp)[0]:.3f}"
              f"  full {fp:.3f} / {maxp}  " + "  ".join(f"{c[:6]} {fr[c]:.4f}" for c in CONDITIONS))
    subprocess.run([sys.executable, "kit/check_submission.py", sub, "--data", pub])

    print("\n== (b) structure discovery on the new train histories")
    for name, mod in (("size_state_search", "method2_sizes.state_search"), ("size_share_probe", "method2_sizes.share_probe"),
                      ("size_mechanism", "method2_sizes.analysis")):
        r = subprocess.run([sys.executable, "-m", mod, "--data", pub], capture_output=True, text=True)
        path = os.path.join(a.out, f"discovery_{name}.txt"); open(path, "w").write(r.stdout + r.stderr)
        print(f"  {name}: {path}\n    " + "\n    ".join(r.stdout.strip().splitlines()[:6]))

    print(f"\n== (c) what changed: {a.baseline} -> {a.data}")
    base, _ = fit(public_dir(a.baseline)); Lb, Ln = learned(base), learned(m)
    for k in Ln:
        same = Lb[k] == Ln[k]
        print(f"  {k}:{'  (unchanged)' if same else ''}\n    before: {Lb[k]}" + ("" if same else f"\n    after:  {Ln[k]}"))
    print("\n== trust flags (new round)")
    for f in flags(Ln):
        print("  -", f)
    json.dump({"baseline": Lb, "new": Ln}, open(os.path.join(a.out, "learned_diff.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
