# Round 2: lowest-overfitting COMBINED entry (Thanos + correlated on/off prior + Momin's cross-machine adjustment)

No fitted number comes from validation: rho (correlated prior, per type) from TRAIN histories; beta = 1.94 (cross-machine adjustment)
= maximum likelihood on TRAIN outcomes. Thanos core = Round 1 code path (adapt=False). Validation (/40): tuning 39.162, audit 39.463,
FULL 39.333 (plain Thanos 39.183; Momin beta 1.94 alone 39.317; combined with beta 2.0: 39.335).

| file | sha256 |
| --- | --- |
| `entry.zip` | 3cb53d28e387b0c58e3605b6cab9b13708190b8e817d952b0924ae33d2e8bdf4 |
| `submission.csv` | PENDING: pinned (numpy 2.4.4 / scipy 1.17.1) fresh-unzip run started 11:43, ~11:54; follow-up commit |

Leakage: Thanos suite ALL 8 PASS; correlated-prior step 7/7 PASS; adjustment 4/4 PASS. Runtime: expected ~11 min (same pipeline measured
at 11.1 min with beta 2.0); measured value pending. Do NOT upload the ZIP without the matching submission.csv from this folder.
