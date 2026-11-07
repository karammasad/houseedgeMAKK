"""Benchmark and verify the standard-library Method 5 model (run after train_final.py).

  python -m method5_simulation.bench --data data/main [--parity 40]

Measures, on ONE CPU thread (pure Python is single-threaded; process CPU time is reported next to wall time):
  fit (load weights) + predict(test), fit + predict(validation), against the 5-minute AGENTS.md budget and the 30-minute
  organizer limit; peak memory; two independent runs -> byte-identical CSV; probability validity; torch parity on N
  validation queries (max |difference|). Writes the candidate CSVs inside method5_simulation/out/.
"""
import argparse, hashlib, os, resource, sys, time
from common.data import public_dir, write_predictions
from method5_simulation.model import Model


def run(pub, split, weights=None):
    t0, c0 = time.perf_counter(), time.process_time(); m = Model(weights) if weights else Model(); m = m.fit(pub); t1 = time.perf_counter()
    pred = m.predict(pub, split); t2, c2 = time.perf_counter(), time.process_time()
    return pred, t1 - t0, t2 - t1, c2 - c0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); ap.add_argument("--parity", type=int, default=40); ap.add_argument("--tag", default="s1")
    a = ap.parse_args(); pub = public_dir(a.data); os.makedirs("method5_simulation/out", exist_ok=True)
    for split in ("test", "validation"):
        W = f"method5_simulation/weights/{a.tag}.json"; pred, tf, tp, cpu = run(pub, split, W); path = f"method5_simulation/out/{split}_{a.tag}.csv"; write_predictions(path, pred)
        pred2, *_ = run(pub, split, W); path2 = path + ".rerun"; write_predictions(path2, pred2)
        h1 = hashlib.sha256(open(path, "rb").read()).hexdigest(); h2 = hashlib.sha256(open(path2, "rb").read()).hexdigest(); os.remove(path2)
        bad = sum(1 for p in pred.values() if min(p) < 0 or abs(sum(p) - 1) > 1e-9 or len(p) != 5)
        print(f"{split:10s}: {len(pred)} queries | fit {tf * 1000:.0f} ms + predict {tp:.1f}s (process CPU {cpu:.1f}s) = {tf + tp:.1f}s "
              f"vs 300s budget | invalid rows {bad} | rerun byte-identical: {h1 == h2} (sha256 {h1[:12]})")
    print(f"peak memory (this process): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20):.0f} MiB")
    wsize = os.path.getsize(f"method5_simulation/weights/{a.tag}.json") / 1024
    print(f"exported weights: {wsize:.0f} KB; imports in model.py: json, math, os, operator + common.data (no numpy/torch)")
    if a.parity:
        import numpy as np, torch
        from common.data import load_split
        from method5_simulation import data as D, cv
        P = __import__("json").load(open(f"method5_simulation/weights/{a.tag}.json")); mode = __import__("json").load(open(f"method5_simulation/weights/{a.tag}.meta.json"))["mode"]; net = cv.Net()
        net.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in P.items() if k != "H"}); net.eval()
        q, seq = load_split(pub, "validation"); worst = 0.0   # `pred` = the validation predictions from the loop above (no extra run)
        for r in q[:a.parity]:
            c = D.machine_cats(seq[r["machine_id"]]); W, S = D.build(c, [r["cutoff"]], r["machine_type"], int(r["signal"]))
            with torch.no_grad():
                p = net(torch.from_numpy(cv.seq_input(W, mode, 0)), torch.from_numpy(S)).numpy()[0]
            worst = max(worst, float(np.abs(p - np.array(pred[r["query_id"]])).max()))
        print(f"torch parity on {a.parity} validation queries: max |pure-python - torch| = {worst:.2e}")


if __name__ == "__main__":
    main()
