"""WARNING: proxy disagrees with real validation on blend selection (RESEARCH_LOG #30); diagnostic only.
High-power evaluation: win-probability Brier on held-out HISTORY ticks (validation + test histories, no targets).
Everything (gate, sizes, stackers, correction) is fit on TRAIN only (splits='train'); examples are the K random ticks (+ masked short
copies) per held-out episode from `Model._features`, 5 independent tick samples. Outcome Brier differences are unbiased for
true-probability squared-error differences, so ~30k paired examples see gains the 48-episode halves cannot."""
import numpy as np
from method4_ml.model import Model
from method4_ml.logit_stack import design, predict as lr_predict, check_layout
PUB = "data/main"

def load_held(m, splits=("validation", "test")):
    rows = []
    base = m.tick_seed
    for bag in range(m.bags):
        m.tick_seed = bag
        for s in splits:
            X, y, _ = m._features(PUB, s)
            rows.append((s, bag, np.asarray(X, float), (np.asarray(y) > 0).astype(float)))
    m.tick_seed = base
    return rows

def candidates(m, X):
    g = X[:, 0]
    tree = np.mean([c.predict_proba(X)[:, 1] for c in m.clfs], axis=0)
    off = np.log(np.clip(g, 1e-4, 1 - 1e-4) / (1 - np.clip(g, 1e-4, 1 - 1e-4)))
    lr = lr_predict(design(X, m.lr_spec), m.beta, off)
    mix = m.blend * tree + (1 - m.blend) * g
    return {"gate": g, "tree_blend(v6)": mix, "lr": lr, "v7": m.lr_weight * lr + (1 - m.lr_weight) * mix}

if __name__ == "__main__":
    m = Model(splits="train").fit(PUB)
    held = load_held(m); X = np.vstack([h[2] for h in held]); y = np.concatenate([h[3] for h in held]); check_layout(X)
    short = X[:, 23] <= 64; tag = np.concatenate([[h[0]] * len(h[2]) for h in held])
    C = candidates(m, X); ref = (np.clip(C["gate"], 1e-6, 1 - 1e-6) - y) ** 2
    print(f"held-out examples {len(y)}  (short {short.mean():.2f}); Brier x1e3, delta vs gate, paired SE")
    for k, p in C.items():
        se = (np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2; d = se - ref
        out = f"{k:16s} {1e3 * se.mean():.3f}  d {1e3 * d.mean():+.3f} ± {1e3 * d.std() / np.sqrt(len(d)):.3f}"
        for nm, msk in (("long", ~short), ("short", short), ("val", tag == "validation"), ("test", tag == "test")):
            out += f" | {nm} {1e3 * d[msk].mean():+.3f}"
        print(out)
