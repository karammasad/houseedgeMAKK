"""Per-machine 1x share among wins, first vs second half of history (train, steady + balanced)."""
from collections import Counter
from common.data import load_split
q, seq = load_split("data/main", "train")
for typ, cut in (("steady", .6), ("balanced", .36)):
    H = Counter(); pairs = Counter()
    for r in q:
        if r["machine_type"] != typ: continue
        s = [v for v in seq[r["machine_id"]] if v > 0]
        if len(s) < 60: continue
        a, b = s[:len(s)//2], s[len(s)//2:]
        sa, sb = sum(x == 1 for x in a) / len(a), sum(x == 1 for x in b) / len(b)
        H[round(sa * 20) / 20] += 1; H[round(sb * 20) / 20] += 1
        pairs[(sa > cut, sb > cut)] += 1
    print(typ, "1x share histogram (halves):", sorted(H.items()))
    print("   (first-half high, second-half high):", sorted(pairs.items()))
