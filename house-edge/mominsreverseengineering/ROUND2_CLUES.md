# What we know about round two (Part B), and what to be ready to adjust

Sources: `kit/RULES.md`, `kit/README.md` (participant guide), `kit/DATA.md`, `data/main/metadata.json`, kit code. Nothing here comes from private files.

## Official statements (quote-level facts)

| clue | where | implication |
| --- | --- | --- |
| "Test histories with changed patterns" | RULES.md | new kit has its own train / validation answers / test histories |
| "Adaptation changes **which temporal relationships recur**" | DATA.md | the lag structures (pairs/singles) differ from main: do not hardcode lag pairs |
| "supplies fresh training and validation examples" | DATA.md | everything can be refit; validation answers (incl. targets.csv) are available to tune on |
| Both rounds contain stable, short-history, drift, shared-change | DATA.md | same four conditions, same flip logic, same short-log layout (40 recent + 24 older) |
| "Machines of the same type can respond differently or oppositely; both win chances and winning amounts can change" | RULES.md | polarity, win rate AND size state keep being in play |
| "Useful patterns can repeat across machines, while their direction differs" | DATA.md | structures shared, polarity per machine (our gate already does this) |
| "Lags up to eight" | RULES.md, metadata | gate lag range stays <= 8 (v8 searches to 12: a superset) |
| "Some relationships gradually reverse; others change together across an episode" | DATA.md | drift = per-machine flip, shared change = episode-wide flip |
| Adaptation has a separate frozen pooled reference, scored out of 40 | RULES.md | ratios are relative to a reference refit on the NEW train data |
| `counts_per_condition` validation 24 each (96 episodes, 12 per condition per half) | metadata.json | the audit half is small and noisy: see CONTROL.md on how to read the +0.15 bar |
| "Public demo seeds are reconstructable; event seeds are private" | metadata.json | the practice demo is not the event generator: do not rely on it |
| Main data may be reused during adaptation; main data mounted at `/main_data` | RULES.md, SUBMISSION_README | possible extra training set, but main patterns may now be WRONG; only use it where the fingerprint says nothing moved |

No clue says the *sizes* mechanism changes. Since sizes were exactly two vectors per type on main, check on day one whether that still holds.

## Day-one checklist when the kit arrives (this order)

1. `metadata.json` says `"round": "adaptation"`; copy `public/` to `data/adaptation/` (never over `data/main/`).
2. **Fallback first**: current pipeline refit on the new train data, checked, uploaded. (`README.md`, step 3-5.)
3. `python3 fingerprint.py --data data/adaptation --out experiments/fp_adapt.json`, then
   `python3 fingerprint.py --diff experiments/fp_main.json experiments/fp_adapt.json`.
   Read what moved: structure prior, cell priors, signal slopes, flip prior, size levels, the size menu, raw payout mix.
4. `python3 probes.py --data data/adaptation` (tuning half only): do the generator restrictions still hold?
   (sizes = exactly 2 vectors per type; cell-logit sd ~0.5-0.7; signal shifts the logit monotonically; no episode sharing.)
5. Retune on the tuning half only. Then reverse-engineer with train + tuning half (targets allowed there).
6. Audit half once per finished idea (CONTROL.md rule). Stop new ideas around 6 AM; submit the best version that passed.

## Which parts of v8 could break (what to watch in the diff)

| piece | learned in fit()? | breaks if |
| --- | --- | --- |
| structure family and prior | learned, family capped at lag 12, pairs + AND pairs | new relationships are outside the family (e.g., triples, other gates) |
| cell priors (Beta per type, signal slope) | learned | rate scale or signal effect moved |
| flip prior (`p_change_short`) | learned | flips happen at different positions or more often |
| size state: coupling to the win gate, tie weight | learned (EM) | size lag is no longer tied to the win-gate lag |
| size menu (2 vectors per type) | learned from history | sizes follow a different number of states |
| tree stacker + logistic correction | trained on the new train/history ticks | selection margin in `round2.py` (0.05) catches a stacker that stops helping |

Rule of thumb: if the diff shows a moved piece, adjust that piece; if everything printed is near-identical, spend the night on the probes, not on re-architecting.
