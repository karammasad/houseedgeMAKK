# TASK: Method 1, Bayesian change-point gate model (owner: Person A)

You own the submission. Read ../AGENTS.md first.

## What exists
- `gate.py`: per machine, Bayesian averaging over structures (single lags and lag pairs 1..8, plus "no gate")
  and one optional polarity flip at tick tau. Win probability = posterior mean of the gate cell's Beta-Bernoulli
  rate. A flip swaps the two cells' rates. Prefix sums make each tau O(1).
- `model.py`: `Model(max_lag=8, max_order=2, a=.5, b=.5, p_change=.5, step=2, tau_min=40, sizes="npmle")`.
  Probabilities = [1 - P(win), P(win) * size distribution from method2_sizes].
- Score now: tuning 55.05, audit 54.79. Tuning loss ratios: stable 0.074, short 0.131, drift 0.090, shared 0.065.
- Already flat (don't retune): a, b, p_change, tau_min, step. All within +-0.15.

## Do, in order (each as a Model setting, default = current behaviour, then evaluate)
1. Upload the current entry (CSV + ZIP from tools/make_entry.py) through the Main form; keep the confirmation.
2. A2 Run-length recursion (Adams & MacKay 2007): allow any number of flips with a constant hazard h per tick.
   Setting `hazard` in {1/100, 1/200, 1/400}; replaces the single tau. Keep if drift or shared improves, stable not worse.
3. A3 Gradual drift: polarity moves linearly between states over [tau, tau+width]; `ramp_width` in {0,10,20,40},
   marginalized. Keep if drift ratio < 0.085.
4. A4 Sticky two-state HMM over polarity (Fox et al. 2011), forward filtering; `switch_prob` in {1e-3,3e-3,1e-2,3e-2}.
5. A5 Episode-shared flip mixture: with prob `p_shared` in {0.2,0.5,0.8} all machines in an episode share one tau.
   Needs per-episode prediction (group queries by episode_id inside predict). Keep if shared ratio < 0.06.
6. A6 Asymmetric learned cell priors: fit Beta priors for the "on" and "off" cells on train by marginal likelihood
   (true win rates cluster near 0.88 and 0.03). Keep if short-history ratio < 0.125.
7. A7 Cell priors by machine_type x signal.

## Done when
Each experiment is a setting, logged in experiments/results.csv, and winners are merged with the team at the
5 PM checkpoint. Final main entry built with tools/make_entry.py and uploaded by 7:30 PM.
