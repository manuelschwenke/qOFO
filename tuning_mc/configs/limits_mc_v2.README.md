# `limits_mc_v2.json` — where each number comes from

Companion to `limits_mc_v2.json` (JSON carries no comments and
`stage_1_search.load_limits` rejects unknown fields, so the provenance lives
here). Superseded predecessor: `limits_mc.json`.

| field | v2 | v1 (`limits_mc.json`) | basis |
|---|---|---|---|
| `corridor_excess_pu` | 1e-4 | 1e-4 | unchanged — calibrated 2026-08-04 from the reference run, at the 1e-4 floor |
| `rho_emp_p95` | 1.5 | 1.5 | unchanged — 25 % margin below the OFO bound of 2. A *declared* margin, not a measurement |
| `settling_s` | 1500.0 | 1500.0 | unchanged — deliberately inactive; see the `ConstraintLimits` docstring |
| `tap_ops_per_h` | **1.25** | 6.0268 | **30 tap operations per day per transformer ÷ 24 h.** Author's decision, 2026-08-14 |
| `tap_reversals_per_h` | **0.25** | 1.2054 | 1.25 / 5, preserving v1's ops:reversals ratio — **a choice, see below** |

## Why `tap_ops_per_h` moved by a factor of 4.8

`6.0268` was never a wear budget. It is
`from_reference(margin=1.5)` applied to one measured trajectory: the 2026-08-13
Thevenin reference made **5 taps in its worst 75-min window** (`v2_gen_trip`,
4.0179 ops/h), and the limit is 1.5 × that. At 24 h it implies **145 taps per
day**, against a stated budget of 30. The metric is already worst-transformer
(not fleet sum), so the conversion is a plain division: 30 / 24 = **1.25/h**.

## Why `tap_reversals_per_h = 0.25` is a choice and not a derivation

The v1 pair `6.0268 / 1.2054` is an exact 5:1 ratio, and the ratio is an
**artefact**, not a rule. Both numbers come from `from_reference` at the same
margin 1.5 on the *same* reference window, which happened to contain 5 tap
operations and 1 reversal:

    tap_ops_per_h       = 1.5 × 4.0179  = 6.0268      (5 taps / 75 min)
    tap_reversals_per_h = 1.5 × 0.80357 = 1.2054      (1 reversal / 75 min)

Note what the second line says: `0.80357/h` is *exactly one tap event in a
75-minute window*, i.e. the quantisation step itself. The v1 reversal limit was
therefore 1.5 quantisation steps — it never had the resolution to distinguish a
hunting controller from a quiet one, which is the defect measured on the 0814
holdout (§11 of `docs/daily_log/08_2026/2026-08-14_lambda_calibration_run_and_thesis_9_3.md`).

Nothing in that provenance dictates a ratio, so nothing in it forbids keeping
5:1 either. **5:1 is retained deliberately**, on two grounds:

* it preserves the one empirical relation available between the two quantities
  (the reference's own 5 ops per reversal), and
* no operator statement distinguishes reversal wear from operation wear on this
  plant, so inventing a second independent budget would be less defensible than
  carrying the measured relation across.

**It is a choice and should be reported as one.** Should a separate hunting
budget ever be stated, this number is the one to replace, and `tap_ops_per_h`
is unaffected.

## Resolution — the reason the whole campaign exists

Taps are integers, so both constraints are quantised at one event per window:

| window | quantisation step | `tap_ops_per_h` limit in steps | `tap_reversals_per_h` limit in steps |
|---|---|---|---|
| 90 min (Tier 1) | 0.667 /h | 1.9 | **0.4** — unresolvable |
| 12 h (Tier 2) | 0.0833 /h | 15.0 | **3.0** — resolved |

At 90 minutes the reversal limit of 0.25/h sits *below* a single reversal, and
the ops limit of 1.25/h sits *between* one tap (0.667/h) and two (1.333/h).
Both constraints are binary there and carry no gradient.

## Two files, and why the Tier-1 search does not use this one

`limits_mc_v2.json` is the **Tier-2 budget**. Applying it to the 90-min Tier-1
bank would reject essentially every candidate before the search starts: the
0814 campaign measured `tap_ops_per_h = 2.007` — three taps in a 90-min window
— in *every row of both λ scans*, which is 61 % above a 1.25/h limit. Phase B's
filter rejects infeasible candidates outright, so the pattern search would poll,
find nothing admissible, shrink `delta` and converge without moving. The
calibration curves would still be produced (`rho_emp_p95` does not depend on
feasibility) but every "feasible" column would read `False` and the search
itself would be dead.

That is not a wear finding. It is the *same* unit error the campaign exists to
remove, applied in the opposite direction: 1.25/h is a **daily average** over a
day that is mostly quiet, and a 90-min window built around an injected
disturbance is not a day. `ConstraintLimits.tap_ops_per_h`'s own docstring puts
the inflation factor at roughly 19×.

So the Tier-1 phases run `limits_mc_v2_tier1.json`, which differs in exactly two
fields:

| field | Tier 1 | Tier 2 | role |
|---|---|---|---|
| `tap_ops_per_h` | 4.0 | 1.25 | 4.0/h = 6 taps per 90-min window = **2× what every 0814 candidate did**. A chatter screen, not a budget |
| `tap_reversals_per_h` | 2.0 | 0.25 | 2.0/h = 3 reversals per window. Rejects sustained hunting without tripping on a single reversal |

Both Tier-1 values sit at least three quantisation steps above zero, so both are
resolvable at 90 min — unlike the pair they replace. Everything else is
identical, including `rho_emp_p95 = 1.5`, which *is* meaningful at 90 min.

**The wear and hunting verdicts are measured on Tier 2 only.** Tier-1
`tap_ops_per_h` / `tap_reversals_per_h` are recorded on every candidate and
reported as diagnostics, never as compliance with the 30-taps/day budget.
