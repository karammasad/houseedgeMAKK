"""Model wrapper with the team interface: Model(**kw).fit(pub) then .predict(pub, split) -> {query_id: [p0, p1, p2, p5, p20]}.

splits="train"  strict inductive: every hyperparameter from TRAIN histories only.
splits="all"    transductive: train + validation + test HISTORIES (pre-cutoff observations only; labels / targets / conditions never read).
Only queries.csv and history.csv are opened (common.data.load_split).
"""
import time
import numpy as np
from common.data import load_split, public_dir
from . import egp


class Model:
    def __init__(self, splits="train", verbose=True, tickset="own", adapt=True, cross=True, flipwin=True):
        """adapt=True (round-2 safe default): XOR triples screened, pair-level keep rule, size-state definition learned,
        plus (cross) neighbour-machine and own-size-state gate inputs and (flipwin) a flip-free window learned from the flip positions.
        adapt=False reproduces the Round 1 submission exactly (pairs only, per-(gate, size lag) keep, size state = payout >= 2)."""
        assert splits in ("train", "all") and tickset in ("common", "own")
        self.tickset = tickset; egp.TICKSET = tickset; self.adapt = adapt; self.cross = cross; self.flipwin = flipwin
        self.splits = ("train",) if splits == "train" else ("train", "validation", "test")
        self.verbose = verbose; self.H = None; self.timing = {}

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def fit(self, pub, machines=None):
        t0 = time.time()
        if machines is None:
            pub = public_dir(pub); machines = []
            for sp in self.splits:
                machines += egp.machines_from_split(*load_split(pub, sp))
        egp.TICKSET = self.tickset
        egp.GATE_ORDER, egp.PAIR_KEEP, egp.SIZE_CHOICES = (3, True, ("big", "win", "one")) if self.adapt else (2, False, ("big",))
        egp.CROSS = egp.BIGGATE = bool(self.adapt and self.cross)
        egp.WINDOW_FRAC = 0.45
        self.log(f"EGP fit on {len(machines)} machines ({'+'.join(self.splits)})")
        self.H = egp.fit_hyper(machines, log=self.log); t1 = time.time()
        egp.fit_flip_prior(machines, self.H, log=self.log)
        if self.adapt and self.flipwin:
            # The population fit assumes long logs are flip-free over their first WINDOW_FRAC. If the learned flip positions say
            # otherwise (5% quantile inside the window), shrink the window below that quantile and refit once.
            q05 = egp.flip_quantile(self.H, .05)
            self.log(f"  flip position 5% quantile {q05:.3f} (window {egp.WINDOW_FRAC:.2f})")
            if q05 < egp.WINDOW_FRAC - .02:
                egp.WINDOW_FRAC = max(.15, q05 - .05)
                self.log(f"  -> refit with flip-free window {egp.WINDOW_FRAC:.2f}")
                self.H = egp.fit_hyper(machines, log=self.log); t1 = time.time()
                egp.fit_flip_prior(machines, self.H, log=self.log)
        self.H.window_frac = egp.WINDOW_FRAC
        self.timing["fit_hyper"] = t1 - t0; self.timing["fit_flip"] = time.time() - t1
        return self

    def predict_machines(self, machines):
        egp.TICKSET = self.tickset; egp.WINDOW_FRAC = getattr(self.H, "window_frac", .45)
        egp.set_size_def(machines, getattr(self.H, "size_def", "big"))
        return {m.qid: egp.machine_posterior(m, self.H)[1].tolist() for m in machines}

    def predict(self, pub, split):
        t0 = time.time()
        out = self.predict_machines(egp.machines_from_split(*load_split(public_dir(pub), split)))
        self.timing[f"predict_{split}"] = time.time() - t0
        return out
