"""Independent test of H-route (LOG.md section 2): EGP vs champion on TRAIN outcomes at cutoffs c - j, j = 0..J (no validation data).
    python -m thanos_method.route_test --ckpt build/main_sb_ckpt/entry_src --hyper <egp_trans>/hyper.pkl --out <dir>
Champion predictions come from its saved checkpoint, fed pseudo-public folders whose train histories are truncated at c - j.
"""
import argparse, csv, os, pickle, shutil, sys, tempfile
from collections import defaultdict
import numpy as np

PAY = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hyper", required=True); ap.add_argument("--out", required=True); ap.add_argument("--J", type=int, default=15); ap.add_argument("--tickset", default="own")
    a = ap.parse_args(); data = os.path.abspath(a.data); os.makedirs(a.out, exist_ok=True)
    from common.data import load_split
    from thanos_method import egp
    H = pickle.load(open(a.hyper, "rb")); egp.TICKSET = a.tickset
    q, seq = load_split(data, "train")
    labels = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(open(os.path.join(data, "train", "labels.csv")))}
    hist = list(csv.DictReader(open(os.path.join(data, "train", "history.csv"))))
    import subprocess
    rows = []
    for j in range(a.J + 1):
        tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "train"))
        cut = {r["machine_id"]: r["cutoff"] - j for r in q}
        with open(os.path.join(tmp, "train", "queries.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"])
            for r in q:
                w.writerow([r["query_id"], r["episode_id"], r["machine_id"], r["machine_type"], r["signal"], cut[r["machine_id"]]])
        with open(os.path.join(tmp, "train", "history.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["episode_id", "machine_id", "tick", "payout"])
            for h in hist:
                if int(h["tick"]) < cut[h["machine_id"]]:
                    w.writerow([h["episode_id"], h["machine_id"], h["tick"], h["payout"]])
        # champion runs in its own process from the checkpoint folder (its packages shadow this repo's method4_ml)
        outp = os.path.join(tmp, "pred.pkl")
        code = ("import pickle,sys; sys.path.insert(0,'.'); m=pickle.load(open('model.pkl','rb')); "
                f"pickle.dump(m.predict({tmp!r},'train'), open({outp!r},'wb'))")
        subprocess.run([sys.executable, "-c", code], cwd=os.path.abspath(a.ckpt), check=True, stdout=subprocess.DEVNULL)
        Pc = pickle.load(open(outp, "rb")); shutil.rmtree(tmp)
        for r in q:
            mid = r["machine_id"]; c = cut[mid]
            y = labels[r["query_id"]] if j == 0 else seq[mid][c]
            if y < 0:
                continue
            m = egp.Machine(r["query_id"], r["episode_id"], mid, r["machine_type"], r["signal"], seq[mid][:c])
            pe = egp.machine_posterior(m, H)[1]
            rows.append(dict(j=j, ep=r["episode_id"], qid=r["query_id"], type=r["machine_type"], short=m.short, y=PAY[y], pe=pe, pc=np.array(Pc[r["query_id"]])))
        print(f"j={j}: {len(rows)} rows", flush=True)
    pickle.dump(rows, open(os.path.join(a.out, "route_rows.pkl"), "wb"))
    report(rows)


def report(rows):
    def brier(p, y):
        oh = np.zeros(5); oh[y] = 1; return float(((p - oh) ** 2).sum())
    rng = np.random.default_rng(0)
    for name, sel in (("LONG logs", lambda r: not r["short"]), ("SHORT logs", lambda r: r["short"]), ("all", lambda r: True)):
        R = [r for r in rows if sel(r)]; by = defaultdict(list)
        for r in R:
            by[r["ep"]].append(brier(r["pe"], r["y"]) - brier(r["pc"], r["y"]))
        d = np.array([np.mean(v) for v in by.values()]); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
        j0 = [brier(r["pe"], r["y"]) - brier(r["pc"], r["y"]) for r in R if r["j"] == 0]
        print(f"{name:10s} rows {len(R):5d} episodes {len(d):3d}  EGP - champion Brier {d.mean():+.6f}  95% CI [{np.percentile(bs, 2.5):+.6f}, "
              f"{np.percentile(bs, 97.5):+.6f}]  P(EGP better) {(bs < 0).mean():.3f}  | real cutoff only {np.mean(j0):+.6f} (n {len(j0)})")


if __name__ == "__main__":
    main()
