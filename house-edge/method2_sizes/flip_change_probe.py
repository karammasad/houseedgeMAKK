"""What changes at a flip besides polarity? Train histories, drift / shared / stable machines (conditions used for
ANALYSIS only).   python -m method2_sizes.flip_change_probe --data data/main

Per long machine: best XOR pair and flip tau from flip_time_probe.scan (stable machines get a placebo tau = 0.69 *
cutoff with no polarity swap). Compares before vs after the flip:
  on/off-cell win rates (after the swap, the 'on' cell is the opposite XOR value)
  P(1x | win) level, and the sign/size of the size-lag effect for the machine's strongest size lag
"""
import argparse, csv, itertools, os
from collections import defaultdict
from common.data import load_split, public_dir, wins
from method2_sizes.flip_time_probe import scan


def cell_rates(w, a, b, lo, hi):
    n = [0, 0]; k = [0, 0]
    for t in range(max(b, lo), hi):
        if w[t] >= 0 and w[t - a] >= 0 and w[t - b] >= 0:
            x = w[t - a] ^ w[t - b]; n[x] += 1; k[x] += w[t]
    return n, k


def size_eff(seq, lo, hi, l):
    n = [0, 0]; k = [0, 0]
    for t in range(max(l, lo), hi):
        if seq[t] > 0 and seq[t - l] >= 0:
            g = int(seq[t - l] > 1); n[g] += 1; k[g] += int(seq[t] == 1)
    return (k[1] / n[1] - k[0] / n[0]) if min(n) >= 6 else None


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/main"); a = ap.parse_args()
    pub = public_dir(a.data); q, seq = load_split(pub, "train")
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(pub, "train", "conditions.csv")))}
    out = defaultdict(lambda: defaultdict(list))
    for r in q:
        c = cond[r["query_id"]]
        if c == "short_history":
            continue
        s = seq[r["machine_id"]]; w = wins(s); cut = r["cutoff"]; best = scan(w, cut)
        if not best or (c != "stable" and best[2] < 8):
            continue
        (aa, bb), tau = best[1], best[3] if c != "stable" else int(.69 * cut)
        n0, k0 = cell_rates(w, aa, bb, 0, tau); n1, k1 = cell_rates(w, aa, bb, tau, cut)
        if min(n0 + n1) < 8:
            continue
        pre = [k0[x] / n0[x] for x in (0, 1)]; on = 0 if pre[0] > pre[1] else 1
        swap = (c != "stable")                    # after a flip the on-cell is the opposite XOR value
        post_on = k1[1 - on] / n1[1 - on] if swap else k1[on] / n1[on]
        post_off = k1[on] / n1[on] if swap else k1[1 - on] / n1[1 - on]
        d = out[c]
        d["d_on"].append(post_on - pre[on]); d["d_off"].append(post_off - pre[1 - on])
        w0 = [v for v in s[:tau] if v > 0]; w1 = [v for v in s[tau:] if v > 0]
        if len(w0) >= 15 and len(w1) >= 15:
            d["d_1x"].append(sum(v == 1 for v in w1) / len(w1) - sum(v == 1 for v in w0) / len(w0))
            ls = max(range(1, 9), key=lambda l: abs(size_eff(s, 0, cut, l) or 0)); e0, e1 = size_eff(s, 0, tau, ls), size_eff(s, tau, cut, ls)
            if e0 is not None and e1 is not None and abs(e0) > .15 and abs(e1) > .15:
                d["size_sign_kept"].append(1.0 if e0 * e1 > 0 else 0.0); d["size_eff_ratio"].append(abs(e1) / abs(e0))
    print(f"{'condition':14s} {'n':>4s} {'d(on rate)':>11s} {'|d on|':>8s} {'d(off rate)':>12s} {'|d off|':>8s} {'d(P1x|win)':>11s} {'|d 1x|':>8s}  size-gate sign kept (n)")
    for c in ("stable", "drift", "shared_change"):
        d = out[c]
        print(f"{c:14s} {len(d['d_on']):4d} {mean(d['d_on']):+11.3f} {mean([abs(x) for x in d['d_on']]):8.3f} {mean(d['d_off']):+12.3f} "
              f"{mean([abs(x) for x in d['d_off']]):8.3f} {mean(d['d_1x']):+11.3f} {mean([abs(x) for x in d['d_1x']]):8.3f}  "
              f"{mean(d['size_sign_kept']):.2f} ({len(d['size_sign_kept'])}), |effect| after/before {mean(d['size_eff_ratio']):.2f}")


if __name__ == "__main__":
    main()
