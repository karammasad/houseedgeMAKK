# Findings

Ledger of hypotheses, tests and verdicts. Newest at the top. Each entry: hypothesis, test, numbers, verdict, why.
Reverts stay here as negative results. See CONTROL.md for the keep rule.

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
