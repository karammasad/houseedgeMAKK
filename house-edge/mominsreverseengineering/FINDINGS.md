# Findings

Ledger of hypotheses, tests and verdicts. Newest at the top. Each entry: hypothesis, test, numbers, verdict, why.
Reverts stay here as negative results. See CONTROL.md for the keep rule.

## Entry 5: sign-tie experiment (Entry 4) RUN: INCONCLUSIVE, underpowered (neither confirmed nor falsified)

`python3 signtie.py --data ../data/main` (train histories only, 192 stable machines, no targets, no audit). Two runs, both reported:

| run | machines identified from first half | main on-cell mean D (95% CI) | N1 decoy | N2 other machine | N3 time-shift | N4 permuted |
| --- | --- | --- | --- | --- | --- | --- |
| pre-registered design (size lag free, threshold 5 nats) | 14 (13 evaluable) | +0.36 [-0.35, +1.22] | -0.07 | +0.23 | +0.04 | +0.35 |
| exploratory follow-up (`--tie-k`: size lag = smaller gate lag) | 11 (10 evaluable) | +0.61 [-0.10, +1.65] | +0.11 | +0.13 | +0.33 [+0.12, +0.56] | -0.02 |

Deviations, both made before any D was seen: the size-gain threshold went from 8 to 5 nats (the first draft identified 2 of 192);
the `--tie-k` run is a second, EXPLORATORY look and is not part of the pre-registered result.
**Verdict under the pre-registered rule: NOT PASSED** (N2 +0.23 is above the +-0.15 band and the CI contains 0). Read honestly:
- The direction and size match the truth-based finding (+0.61 vs the +0.7 to +0.9 logit gap seen in the tuning-half truth), but with 10 machines
  the CI is [-0.10, +1.65]: it cannot separate +0.6 from 0.
- The controls are not clean nulls at this n: N4 (an estimator null) gave +0.35 in one run and -0.48 / +0.44 in balanced / steady in the other,
  and the time-shift control is +0.33 with a CI above 0 (a persistent state or lag echo is possible). Per-machine D is too noisy here.
- Side result that DID hold: the size lag equals the smaller gate lag for 11 of 11 identified machines in the tied run, and the pairs
  recovered are again exactly (1,3), (2,5), (4,7).
- **The bottleneck is identifying the size state, not the win-rate shift.** Only 6-7% of machines get a decisive state from about 45 wins;
  short logs have 15-20. That argues the payoff of the proposal is smaller than the +0.05 to +0.1 guess in Entry 4: the shift only helps a
  machine whose state direction is known, and a hard threshold understates what a soft posterior extracts (Momin: short-log state is wrong 26%).
Next step, if pursued (not run): pool all 576 long machines with SOFT size-state weights from the existing size model, instead of a hard
threshold on 192; that is a model-based test, so it is no longer training-only-and-model-free.

## Entry 3: Candidate A (gate capped at lag 8): NO-OP (recorded, not adopted)

| run | tuning | audit | full | stable | short | drift | shared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v8 baseline | 58.3107 | 58.2885 | 58.2981 | .01704 | .05671 | .02242 | .02060 |
| A: `gate.max_lag = 8` | 58.3107 | 58.2885 | 58.2981 | .01704 | .05671 | .02242 | .02060 |

Identical in every column. Cause (read from the cached gate fits of both runs): after the learned-prior pruning (`prune=0.01`) the
gate keeps exactly **7 structures**, with 0.0000 prior mass on any lag above 7, for `max_lag` 12 and for 8 alike:
`xs(1,3,1) .183, xs(2,5,2) .169, xs(4,7,4) .158, xor(1,3) .156, xor(4,7) .147, xor(2,5) .140, none .048`.
So the lag-pair support (three pairs, each plain and size-split) is already INFERRED from train, and the official "lags up to eight"
statement adds nothing. Verdict: no change; no further lag-range sweeps.

## Entry 4: map of v8's generative assumptions, and the one missing mechanism (analysis only, nothing implemented)

**Fixed in code** (changeable only by editing config or code): win bit = payout > 0 (1x counts as a win); a gate is the XOR (or one of
4 AND-masks) of at most two win bits; candidate pool = all pairs to lag 12; ONE optional polarity flip per machine that swaps the two
cells and nothing else; flip time uniform on a grid (step 2, from tick 40 to cutoff-3), flip probability .5 on long logs and a learned
.05 on short logs; "on"/"off" assignment averaged 50/50; each cell rate constant in time with an INDEPENDENT Beta prior per cell;
a "short" log = more than half the ticks missing; a tick whose gate inputs are missing is dropped from cell counts, and the cutoff cell
is mixed by observed share when its inputs are missing; size split uses the payout category (loss / 1x / big) at the SMALLER lag only.
**Inferred from data in `fit()`**: which structures exist (330 candidates pruned to 7), their prior (flip-aware EM), Beta cell priors
per type with a learned log-odds slope in `signal`, short-log flip prior, size-state levels, win-pair-to-size-lag coupling and tie
weight, the exact 2-vector size menu per type, the tree stacker and logistic correction.
**Changeable without new algorithm code**: `GATE` and `SIZES` dicts, `size_split` mode, `cell_group`, blend weights, prune/floor.
**Needs new code**: any dependence between sub-cell rates, a joint win x size likelihood, more than one flip.

**The missing mechanism: one hidden binary state drives BOTH the size vector and the win logit, with a type-independent sign.**
Tuning-half truth (`probes.py --probe state`, audit half not read): in every one of the six type x gate-mode cells the low-1x (bigger
payout) size state has the HIGHER win rate: balanced off .081 vs .035, on .890 vs .792; jackpot off .021 vs .009, on .700 vs .501;
steady off .209 vs .117, on .962 vs .941. In logit units the gap is about 0.7-0.9 in five of the six cells (steady on 0.46). The state
explains 18-40% of the within-(type, mode) variance of true P(win) (37% pooled; 68% on short-history episodes, n too small to trust).
The model uses `xs`/`xst` cells, whose six (or four) sub-cell rates have EXCHANGEABLE Beta priors: same mean, no sign, no link to the
size evidence. A short log leaves roughly 5 observations per sub-cell, so the model cannot tell which sub-cell rate applies at the cutoff.
Partial post-hoc capture exists (stacker feature, logistic correction terms for size state), but not inside the Bayesian posterior.

**Proposal.** Hypothesis H = (pair (i,j), size lag k, state grouping g), direction d, ω on/off assignment, flip time s.
State h_t = 1[category(c_{t-k}) in group g]; gate bit u_t = w_{t-i} xor w_{t-j} xor 1[t >= s] xor ω.
Win: w_t | (u_t, h_t) ~ Bernoulli(ρ_{u h}), logit ρ_{u h} = a_u + γ_type · σ_{h xor d}, with σ = +1 for the larger-expected-size state,
a_u ~ existing on/off prior (with signal slope), γ_type the learned shared shift. Size: y_t | win, h_t ~ Cat(V_{type, h_t xor d}), V known.
Joint posterior: P(H,d,s,ω | w,y) ∝ P(H) P(s) · ∫ Π_u π(a_u) Π_h Bin(k_{uh}; n_{uh}, ρ_{uh}) da · Π_{t: w_t = 1} V_{h_t xor d}(y_t).
Closed-form version: give each sub-cell a Beta with mean shifted ±γ on the log-odds scale and the same strength (adi's `shift_cellprior`
already does this arithmetic). γ = 0 and the size factor off reproduce the present model exactly (nested). Prediction:
p(0) = 1 - E[ρ_{u_c h_c}], p(z) = E[ρ_{u_c h_c} · V_{h_c xor d}(z)] under the joint posterior (this also removes the product-of-marginals bias).
Learning: γ_type by train marginal likelihood on a grid (the same coordinate ascent as the signal slope), size factor on/off by train evidence.
Cost (estimate, not timed): learning +10-20% of the signal-slope step; prediction unchanged in order (7 structures, closed form);
tens of seconds to a few minutes on a laptop. Not implemented.

**Experiment (training only, negative control; not yet run).** On the stable train machines (192; condition labels select the
analysis subset only, never enter a model), split each history in half. From the FIRST half only: gate pair by MAP, orientation of on/off,
and each tick's size state from the existing size model. In the SECOND half, in on-cell ticks, compute D_m = logit(win rate | big-size state)
- logit(win rate | small-size state) (add-half smoothing). Prediction: mean D > 0 by about 0.5-0.9, sign agreement well above 55% in at
least two types. Controls: (N1) define the state from a decoy lag outside {i, j, k}; (N2) permute the first-half size-state labels across
machines of the same type. Both must give mean D inside +-0.15 and sign agreement near 50%. Falsified if the main D is inside +-0.15,
or either control matches it. If it passes, the keep-check protocol decides adoption.

## Entry 1: generator-constraint probes on main (tuning half only; diagnosis, no model change)

Tool: `python3 probes.py --data data/main` (48 tuning episodes, 384 queries; audit half not read). Nothing here is a model
change; each line is a hypothesis about how restricted the generator is, to be re-checked on the round-two kit.

| hypothesis (from the pasted threat list) | result on main tuning half | status |
| --- | --- | --- |
| Machine types share exact parameter constraints (sizes) | P(size given win) takes exactly 2 vectors per type: balanced [.500 .348 .130 .022] / [.214 .547 .205 .034], jackpot [.335 .245 .315 .105] / [.121 .324 .416 .139], steady [.455 .436 .107 .002] / [.754 .197 .048 .001]. 100% of queries on one of the two. | HOLDS. Already exploited by v8's size menu (learned from histories). Re-check on round two (day-one step 4). |
| Win probabilities come from a small discrete menu | True P(win) is continuous: 85-98% of values distinct in 8 of 9 type x signal cells (jackpot signal -1: 21 of 36, because probabilities near .003 collapse at 3 decimals). | FALSIFIED (agrees with Kareem's #34). |
| Signal encodes a hidden parameter more directly than we model | Signal does NOT predict the size state (hi-1x share 45-65% for every signal). Overall P(win) eta^2 for signal = .003. BUT inside a gate mode the logit of the win rate shifts monotonically with signal, about +0.3 to +0.55 logits per step in every type and mode. | PARTLY. The shift is real and already learned (signal slopes). A single shared slope across types is untested. |
| Hidden rate structure inside the on/off modes | Within-mode logit sd is nearly the same everywhere: balanced off .73 / on .65, jackpot off .72 / on .70, steady off .60 / on .53. | NEW LEAD (see below). |
| Episode-level sharing of rates | Episode eta^2 .120 vs chance level .125 (48 groups / 384). | No sharing (agrees with the ICC .07-.09 finding). |

**Lead from the within-mode spread.** Cell rates look like logit-normal: logit(rate) = base[type, mode] + beta * signal + N(0, sigma^2)
with sigma about 0.6-0.7 and beta about 0.4, close to shared across types. Our gate uses Beta priors per type with separate
per-type slopes. If the generator really is logit-normal with shared sigma and beta, a logit-normal prior (grid quadrature) with
shared sigma/beta, both learned in `fit()`, gives sharper priors on short logs. Cost: the gate's cell evidence is closed-form
Beta-Binomial, so this is a real code change in the gate. Prior art: mixture-of-Betas gave nothing (method4 #9), so expect a small gain.
**Tested (`probes.py --probe shape`, true cell rates on the tuning half, 18 type x mode x signal cells, 2 parameters each):**
logit-normal beats Beta by +5.8 log-likelihood units over 384 values (+0.015 per value); per cell the difference is -0.30
to +0.95. That is indistinguishable at this n, and these are the TRUE rates, a cleaner test than anything the gate sees.
**Verdict: do not rewrite the gate prior.** Beta is an adequate description of the cell rates. The only residual use of the
finding is the near-constant within-mode logit sd (~.6-.7) as a sanity check on the round-two kit (a changed sd would show up
in the fingerprint's cell-prior `k` values).

## Entry 0: baseline (v8 as copied)

- Hypothesis: none. This is the reference every later keep check compares against.
- Test: `python3 run.py score --name "v8 baseline"` on `../data/main`.
- Expected (from Kareem's log, e46c44d): tuning 58.309, audit 58.287, full 58.297. Verdict: reference.
