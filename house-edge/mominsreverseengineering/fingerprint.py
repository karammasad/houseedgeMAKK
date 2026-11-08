"""Generator fingerprint: what the model learned about a kit's mechanism, from train + HISTORIES only (no targets).

  python3 fingerprint.py --data data/main       --out experiments/fp_main.json
  python3 fingerprint.py --data data/adaptation --out experiments/fp_adapt.json
  python3 fingerprint.py --diff experiments/fp_main.json experiments/fp_adapt.json

Why: round two "changes which temporal relationships recur". The diff shows WHICH learned pieces moved (structure prior,
cell priors, signal slopes, flip prior, size levels, the 2-vector size menu, raw payout mix), so we adjust only what moved.
Reads no labels/targets/conditions; safe to run on the new kit the moment it arrives. Fits the gate + size model + menu only
(no stacker), so it takes about as long as one cached gate fit.
"""
import argparse, json, os, sys
from collections import defaultdict
from types import SimpleNamespace
from common.data import public_dir, load_split
from method4_ml.model import Model
from method4_ml.round2 import report
from method4_ml.size_menu import fit_menu


def raw_stats(pub):
    """Model-free description of the kit: payout mix, win rate by type x signal, missing share, cutoff range."""
    out = {}
    wins = defaultdict(lambda: [0, 0]); mix = defaultdict(lambda: defaultdict(int)); miss = [0, 0]; cuts = []
    for sp in ("train", "validation", "test"):
        if not os.path.exists(os.path.join(pub, sp, "history.csv")):
            continue
        q, seq = load_split(pub, sp)
        for r in q:
            cuts.append(r["cutoff"])
            for v in seq[r["machine_id"]]:
                miss[1] += 1
                if v < 0:
                    miss[0] += 1; continue
                k = (r["machine_type"], r["signal"]); wins[k][1] += 1; wins[k][0] += v > 0
                if v > 0:
                    mix[r["machine_type"]][v] += 1
    out["win_rate type|signal"] = {f"{t}|{s}": round(w / max(n, 1), 4) for (t, s), (w, n) in sorted(wins.items())}
    out["payout mix given win"] = {t: {str(p): round(c / max(sum(d.values()), 1), 4) for p, c in sorted(d.items())} for t, d in mix.items()}
    out["missing share"] = round(miss[0] / max(miss[1], 1), 4)
    out["cutoff range"] = [min(cuts), max(cuts)]
    return out


def fingerprint(data, menu_k=32):
    pub = public_dir(data)
    m = Model(cache=True)
    gm = m._fit_gate(pub); g = m._fit_sizes(pub, gm)
    shim = SimpleNamespace(gm=gm, size_models={"gate": g})
    rep, flags = report(shim)
    menu = fit_menu(pub, ("train", "validation", "test"), g, menu_k, 1)
    rep["size menu (V1, V2 per type)"] = {t: [[round(x, 3) for x in v] for v in vs] for t, vs in menu.items()}
    rep["raw"] = raw_stats(pub)
    rep["flags"] = flags
    with open(os.path.join(pub, "metadata.json")) as f:
        meta = json.load(f)
    rep["kit"] = {k: meta.get(k) for k in ("round", "season_id", "suite_version", "history")}
    return rep


def flatten(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(x, list) and x and all(isinstance(i, list) and len(i) == 2 and isinstance(i[0], str) for i in x):
        for name, p in x:                                    # [(name, prob), ...] pairs -> keyed by name
            out[f"{prefix}/{name}"] = p
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = x
    return out


def diff(a, b, abs_tol=0.05, rel_tol=0.25):
    fa, fb = flatten(a), flatten(b); rows = []
    for k in sorted(set(fa) | set(fb)):
        if k.startswith("/kit") or k.startswith("/flags"):
            continue
        va, vb = fa.get(k), fb.get(k)
        if va is None or vb is None:
            rows.append((k, va, vb, "ONLY IN " + ("B" if va is None else "A"))); continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = abs(va - vb)
            if d > abs_tol and d > rel_tol * max(abs(va), abs(vb)):
                rows.append((k, va, vb, f"{vb - va:+.3f}"))
        elif va != vb:
            rows.append((k, va, vb, "changed"))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--diff", nargs=2)
    a = ap.parse_args()
    if a.diff:
        A, B = (json.load(open(p)) for p in a.diff)
        rows = diff(A, B)
        print(f"A: {A.get('kit')}\nB: {B.get('kit')}\n{len(rows)} learned values moved (> .05 abs and > 25% rel):")
        for k, va, vb, note in rows:
            print(f"  {k:70s} {va!s:>12} -> {vb!s:<12} {note}")
        print("\nflags A:", *A.get("flags", []), sep="\n  "); print("flags B:", *B.get("flags", []), sep="\n  ")
        return
    if not a.data or not a.out:
        sys.exit("need --data and --out (or --diff A B)")
    rep = fingerprint(a.data)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k != "raw"}, indent=1, default=float)); print("FLAGS:", *rep["flags"], sep="\n  ")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
