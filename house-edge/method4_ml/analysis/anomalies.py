"""Data audit across train / validation / test: shifts, duplicates, degenerate histories, tick structure."""
import csv, statistics as st
from collections import Counter, defaultdict
from common.data import load_split, PAYOUTS
PUB = "data/main"
for sp in ("train", "validation", "test"):
    q, seq = load_split(PUB, sp)
    rows = list(csv.DictReader(open(f"{PUB}/{sp}/history.csv")))
    dup = len(rows) - len({(r["machine_id"], r["tick"]) for r in rows})
    badpay = Counter(r["payout"] for r in rows if int(r["payout"]) not in PAYOUTS)
    beyond = sum(int(r["tick"]) >= next(x["cutoff"] for x in q if x["machine_id"] == r["machine_id"]) for r in rows[:2000])
    cuts = [r["cutoff"] for r in q]; obs = [sum(v >= 0 for v in seq[r["machine_id"]]) for r in q]
    short = sum(o <= 64 for o in obs)
    pay = Counter(v for r in q for v in seq[r["machine_id"]] if v >= 0); tot = sum(pay.values())
    consts = sum(len({v for v in seq[r["machine_id"]] if v >= 0}) == 1 for r in q)
    zero_wins = sum(sum(v > 0 for v in seq[r["machine_id"]]) < 3 for r in q)
    eps = Counter(r["episode_id"] for r in q)
    ep_cut = defaultdict(set)
    for r in q: ep_cut[r["episode_id"]].add(r["cutoff"])
    tm = Counter(r["machine_type"] for r in q); sg = Counter(r["signal"] for r in q)
    print(f"== {sp}: {len(q)} queries, {len(eps)} episodes (sizes {sorted(set(eps.values()))}), cutoff {min(cuts)}-{max(cuts)} mean {st.mean(cuts):.0f}; "
          f"episodes w/ >1 cutoff: {sum(len(v) > 1 for v in ep_cut.values())}")
    print(f"   dup ticks {dup}, bad payouts {dict(badpay)}, rows>=cutoff(first 2000) {beyond}, short(<=64 obs) {short} ({short / len(q):.0%}), constant histories {consts}, <3 wins {zero_wins}")
    print("   payout mix", {k: round(pay[k] / tot, 3) for k in PAYOUTS}, "types", dict(tm), "signals", dict(sg))
    # tick structure of short histories: where are the 24 sampled ticks?
    older = []
    for r in q:
        s = seq[r["machine_id"]]; o = [t for t, v in enumerate(s) if v >= 0]
        if len(o) <= 64: older += [r["cutoff"] - 1 - t for t in o[:24]]
    if older: print(f"   short-history older ticks: age min {min(older)} median {st.median(older):.0f} max {max(older)}; recent block starts at cutoff-{40}")
    # win rate by tick position
    early = st.mean(int(v > 0) for r in q for v in seq[r["machine_id"]][:50] if v >= 0)
    late = st.mean(int(v > 0) for r in q for v in seq[r["machine_id"]][-50:] if v >= 0)
    print(f"   win rate first 50 ticks {early:.3f}, last 50 ticks {late:.3f}")
