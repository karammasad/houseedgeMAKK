"""Offset-logistic correction of the gate's P(win), fit by L2-penalized Newton (IRLS). Owner: Person D. numpy only.
Idea from adi round 7 (adi/sizewin): keep the gate's logit as a FIXED offset (a free slope hurt: it came out .93) and learn a
small structured correction. logit P' = logit P + Z @ beta, with Z built from my feature matrix columns (see COL below)."""
import numpy as np

# column layout of features3.episode_examples: g(7) lg ep(4) base(13) type signal [size a, a - mu]
COL = dict(pw=0, pw_nc=1, p_chg=2, p_none=3, top1=4, top2=5, rec=6, lg=7, ep_chg=8, ep_chgmax=9, ep_rec=10, ep_shift=11,
           lag1=12, rate10=20, rate20=21, rate40=22, nobs=23, miss40=24, type=25, signal=26, a=27, adev=28)


NBASE = 29   # columns of features3 before the optional polarity bank (polarity.NFEAT columns, appended last)
POL = dict(nb_z=29, nb_sh=30, sz_z=33, sz_sh=34, nx_z=37, nx_sh=38)


def check_layout(X, ntypes=3):
    """Fail loudly if the column layout drifts from features3 (a miscounted index silently ruined an earlier experiment).
    Types and signals are whatever the round uses (learned in fit), so only their form is checked."""
    from method4_ml.polarity import NFEAT
    assert set(np.unique(X[:, COL["type"]])) <= set(map(float, range(ntypes))), "type column wrong"
    sg = X[:, COL["signal"]]; assert np.all(np.isfinite(sg)) and np.all(sg == np.round(sg)), "signal column wrong"
    assert np.all((X[:, COL["pw"]] >= 0) & (X[:, COL["pw"]] <= 1)), "pw column wrong"
    assert np.all(X[:, COL["nobs"]] >= 0) and np.all(X[:, COL["nobs"]] == np.round(X[:, COL["nobs"]])), "nobs column wrong"
    assert np.all(np.abs(X[:, COL["adev"]]) < 1) and np.all((X[:, COL["a"]] >= 0) & (X[:, COL["a"]] <= 1)), "size columns wrong"
    assert X.shape[1] in (NBASE, NBASE + NFEAT), f"expected {NBASE} or {NBASE + NFEAT} features, got {X.shape[1]}"


def design(X, spec, ntypes=3):
    """Columns per spec. 'd' = P(1x|win) deviation from type mean; P = gate P(win); typed terms get one column per machine type.
    'pol' (needs the polarity bank): each bank's best predicted shift, weighted by how far its evidence is above noise."""
    P = np.clip(X[:, COL["pw"]], 1e-4, 1 - 1e-4); d = X[:, COL["adev"]]; ty = X[:, COL["type"]].astype(int)
    short = (X[:, COL["nobs"]] <= 64).astype(float); sig = X[:, COL["signal"]]
    cols = []
    def typed(v):
        for t in range(ntypes): cols.append(v * (ty == t))
    if "d" in spec: typed(d * P); typed(d * (1 - P))                      # adi's 6-term correction
    if "dshort" in spec: typed(short * d * P); typed(short * d * (1 - P))
    if "chg" in spec: typed(X[:, COL["p_chg"]] * (2 * P - 1))
    if "curv" in spec: typed((P - .5) * np.abs(P - .5))
    if "sig" in spec: typed(sig * P); typed(sig * (1 - P))
    if "ep" in spec: cols += [X[:, COL["ep_shift"]], X[:, COL["ep_chg"]] * (2 * P - 1)]
    if "short" in spec: typed(short * (2 * P - 1))
    if "pol" in spec and X.shape[1] > NBASE:
        for b in ("nb", "sz", "nx"):
            cols.append(X[:, POL[b + "_sh"]] * np.clip(X[:, POL[b + "_z"]] / 2 - 1, 0, 2))
    return np.column_stack(cols) if cols else np.zeros((len(X), 0))


def fit_offset_logistic(Z, y, offset, l2=1.0, iters=30, w=None):
    """beta = argmin -loglik(y | sigmoid(offset + Z beta)) + l2 * |beta|^2 (no intercept: the offset is fixed)."""
    beta = np.zeros(Z.shape[1]); y = np.asarray(y, float); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(offset + Z @ beta))); g = Z.T @ (w * (y - p)) - 2 * l2 * beta
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + 2 * l2 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g); beta += step
        if np.max(np.abs(step)) < 1e-8: break
    return beta


def predict(Z, beta, offset):
    return 1 / (1 + np.exp(-(offset + Z @ beta)))
