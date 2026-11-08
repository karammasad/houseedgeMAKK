"""Export method4_ml v5's trained gradient-boosted stacker to plain JSON (run OFFLINE, needs scikit-learn).

Why: method4_ml v5 (Kareem) is the team's most accurate model but needs scikit-learn at prediction time, which the
team won't ship (AGENTS rule 4). Its features, gate and size models are standard library; only the trees are not.
This script trains v5 exactly as method4_ml does, then writes every tree (all seeds and iterations) as node tables
to adi/stack/trees_<round>.json. adi/stack/model.py evaluates them in pure Python, so the entry ships the trained
trees as a checkpoint (allowed by the participant guide) and runs with the standard library only.

  /path/to/venv-with-sklearn/bin/python -m adi.stack.export --data data/main --out adi/stack/trees_main.json

Also writes scikit-learn's own validation/test win probabilities next to the JSON (".sk_check.json") so
adi.stack_model can verify the pure-Python trees reproduce them.
"""
import argparse, json, os, sys


def export_clf(clf):
    trees = []
    for it in clf._predictors:
        (pred,) = it  # binary classification: one tree per iteration
        nodes = []
        for n in pred.nodes:
            nodes.append([int(n["is_leaf"]), int(n["feature_idx"]), float(n["num_threshold"]),
                          int(n["missing_go_to_left"]), int(n["left"]), int(n["right"]), float(n["value"])])
        trees.append(nodes)
    return {"baseline": float(clf._baseline_prediction.ravel()[0]), "trees": trees}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/main"); ap.add_argument("--out", required=True)
    ap.add_argument("--kwargs", default="{}", help="method4_ml.model.Model settings (default = v5 defaults)")
    a = ap.parse_args()
    import sklearn
    from common.data import public_dir
    from method4_ml.model import Model as M4
    pub = public_dir(a.data)
    kw = dict(json.loads(a.kwargs)); kw["cache"] = False
    m = M4(**kw).fit(pub)
    if m.stack != "hgb" or m.sclfs:
        sys.exit("only the default HGB win stacker (no size stacker) is supported")
    out = {"sklearn": sklearn.__version__, "m4_kwargs": kw, "blend": m.blend, "n_train": m.n_train,
           "models": [export_clf(c) for c in m.clfs]}
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    check = {}
    for split in ("validation", "test"):
        X, _, ids = m._features(pub, split, queries=True)
        Xa = m._sel(X); P = [0.0] * len(ids)
        for c in m.clfs:
            for i, p in enumerate(c.predict_proba(Xa)[:, 1]):
                P[i] += p / len(m.clfs)
        check[split] = dict(zip(ids, P))
    with open(a.out + ".sk_check.json", "w") as f:
        json.dump(check, f)
    print("wrote", a.out, os.path.getsize(a.out), "bytes;", sum(len(x["trees"]) for x in out["models"]), "trees")


if __name__ == "__main__":
    main()
