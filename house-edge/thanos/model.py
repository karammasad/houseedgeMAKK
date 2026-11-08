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
    def __init__(self, splits="train", verbose=True, tickset="own"):
        assert splits in ("train", "all") and tickset in ("common", "own")
        self.tickset = tickset; egp.TICKSET = tickset
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
        self.log(f"EGP fit on {len(machines)} machines ({'+'.join(self.splits)})")
        self.H = egp.fit_hyper(machines, log=self.log); t1 = time.time()
        egp.fit_flip_prior(machines, self.H, log=self.log)
        self.timing["fit_hyper"] = t1 - t0; self.timing["fit_flip"] = time.time() - t1
        return self

    def predict_machines(self, machines):
        egp.TICKSET = self.tickset
        return {m.qid: egp.machine_posterior(m, self.H)[1].tolist() for m in machines}

    def predict(self, pub, split):
        t0 = time.time()
        out = self.predict_machines(egp.machines_from_split(*load_split(public_dir(pub), split)))
        self.timing[f"predict_{split}"] = time.time() - t0
        return out
