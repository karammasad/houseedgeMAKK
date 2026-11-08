"""PROPOSED smallest adaptation (a wrapper in OUR folder; Kareem's code is imported and monkeypatched, never edited): guarantee a no-gate candidate in Thanos' structure support.
egp.screen returns the kept (gate, size lag, mass) list; if no no-gate candidate survives (it never does when a constant-gate-bit machine ties every gate with 'none'),
append ((), 1, 0.05). Nested: the EM in fit_hyper then sets its prior mass from the data (smoothing floor ~1e-2/N), so mass ~0 recovers the original model.
Use:  import nogate_fix; nogate_fix.apply(egp)  BEFORE Model.fit."""
def apply(egp, mass=0.05, size_lag=1):
    orig = egp.screen
    def screen_fix(machines, min_mass=0.01, iters=60):
        keep, P = orig(machines, min_mass=min_mass, iters=iters)
        if not any(len(G) == 0 for G, _, _ in keep): keep = keep + [((), size_lag, mass)]
        return keep, P
    egp.screen = screen_fix; return egp
