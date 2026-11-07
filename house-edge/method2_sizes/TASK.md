# TASK: Method 2, payout sizes given a win (owner: Person B)

Read ../AGENTS.md first.

## What exists
- `sizes.py`: `DirichletSizes(strength)` (type x signal prior) and `NPMLESizes(min_wins=30, iters=300)`
  (Kiefer-Wolfowitz NPMLE prior over machine size distributions per machine type, EM weights; Koenker & Gu).
- `model.py`: Method 1's gate with a size model chosen by the `sizes` setting. Add new size models to sizes.py
  and a new `sizes=` option in method1_gate/model.py's constructor (tell Person A).
- Score: NPMLE 55.05 tuning / 54.79 audit; Dirichlet(30) 54.88 / 54.56.
- Upper bound with TRUE sizes: about +2.5 points. Known: size depends on machine type, NOT signal; machines differ
  mostly in the 1x vs 2x split; size does not depend on the gate cell; last-40-tick size counts are worse.

## Do, in order
1. B1 Mechanism test: in train histories, tabulate the size of a win by the previous win's size and by each lag's
   payout (lags 1..8). Write findings to FINDINGS.md. If a dependence exists, make it a covariate.
2. B2 Episode-wide size shifts: add each episode's recent size deviation from type averages (other machines,
   ticks < cutoff only). Window {20,40,80}, shrinkage {0.25,0.5,1}. Keep if shared-change improves.
3. B3 Size change point: weight size counts by the gate's flip posterior, or apply the change-point model to sizes.
4. B4 Recency weights inside the NPMLE likelihood: half-life {64,128,256}.
5. B5 NPMLE atoms from all splits' histories (allowed: histories only) plus Dirichlet draws; min_wins {20,30,60}.
6. B6 Control: Dirichlet-multinomial with strength fitted per type by Minka's fixed-point iteration.

To measure size changes alone, keep the gate fixed (same Model settings) and change only `sizes`.

## Status (2026-11-07)
- Done: B1 (lag-1 size covariate, `sizes="lag"`), B4 (`npmle_half_life`), B5 (`npmle_min_wins`: no effect),
  B6 (`sizes="minka"`: worse). B2 skipped (episode ICC ≈ 0). B3 blocked (needs the gate's flip posterior from Person A).
- Candidate: `{"sizes":"lag","lag_strength":100,"npmle_half_life":64}` → tuning 55.227 (+0.173), audit 54.851 (+0.056),
  full validation via kit checker 55.044 (+0.117). Details in FINDINGS.md; outputs in out/.
- UPDATE: size gate found (see top of FINDINGS.md). New candidate `{"sizes":"gate","gate_cell_prior":"npmle"}` →
  tuning 56.037 (+0.98), audit 56.377 (+1.58), full validation 56.191 (+1.26). Supersedes the lag candidate.
