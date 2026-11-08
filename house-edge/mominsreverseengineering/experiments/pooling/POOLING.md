# Shared flip-time pooling: oracle ceiling (shared_change episodes, train only) -- CLOSED for Round 1
Setup: 24 shared_change train episodes, 192 machines, 7,512 pseudo-queries (targets 30%-100% of the log), gate model gm_sb, no refit. Train has NO true flip times, so the "oracle" tau_ep is the median of the 8 machines' full-history flip estimates (within-episode std ~1 tick): generous, it uses ticks after the target. "Pooled" reweights each machine's own (structure, tau) posterior by the other machines' tau-likelihood ratios from ticks BEFORE the target only. Five-category squared error x1e-4 vs realized payout, paired, 95% CI over the 24 episodes.
| rows (ticks from the flip) | n | production | oracle - prod | pooled - prod |
|---|---|---|---|---|
| before flip | 4,056 | 3508 | -28.6 [-44.7,-16.6] | -28.6 [-44.7,-16.2] |
| 0-10 after | 480 | 6700 | -1886 [-2492,-1353] | -1012 [-1538,-544] |
| 10-20 after | 480 | 3962 | -12.2 [-28.0,-0.6] | -11.8 [-26.2,-0.3] |
| 20-40 after | 960 | 3454 | -0.5 [-5.4,+3.4] | -1.8 [-4.8,+0.9] |
| QUERY-LIKE (>=38 after) | 1,632 | 3652 | +0.4 [-2.0,+2.9] | -0.7 [-2.3,+1.0] |
Real queries sit at the cutoff, >= 0.23 L (>= ~50 ticks) after the flip in shared_change (tau/L max .77), i.e. in the query-like row: oracle gain +0.4 [-2.0,+2.9], pooled -0.7 [-2.3,+1.0]: zero. Machine level (192): 34% improved >1e-4, 32% worsened; episode level range [-10,+16]. By type all CIs include 0.
All the gain is in the first 10 ticks after the flip (28% of the error there; pooling recovers ~54% of it, from other machines' prefixes). Not reachable on Round 1 query positions.
Stacker episode features (item 4): NOT testable cheaply (stacker rows for train pseudo-queries are not cached; rebuilding features3 episode_examples per tick is the full pipeline). Moot for Round 1 given the zero ceiling. Relevant only if a Round 2 kit puts shared flips within ~10-20 ticks of the cutoff; then pooling would be adaptation insurance worth ~half the near-flip loss.
VERDICT: NO-GO for Round 1; direction closed. Contingency only: check tau/L of the Round 2 kit's train machines (cheap, no labels); if shared flips fall within ~20 ticks of the cutoff, the smallest pooling experiment is this script moved into the gate predict path (per-episode reweighting by the other machines' tau-likelihood ratios; ~1-2 h, negligible runtime).
