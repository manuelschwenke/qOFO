# Switched-shunt banks: mutual exclusion of MSC and MSR per STS

Timestamp: 2026-10-22, Europe/Berlin.

Reason: the closed-loop check after removing the one-step-per-zone and
hard-voltage-row restrictions (`2026-10-22_shunt_bank_restrictions_removed.md`)
showed MSC/MSR pairs standing engaged against each other. At 1800 s the MSCs
engaged at 720 s were inside their holding time and could not be released, so
the MIQP reduced injection by engaging the MSRs of the same STSs, and each of
DSO_1–3 carried one engaged MSC and one engaged MSR for the remaining 5 h.
Author decision: forbid this as a bound, per STS.

## Changes

- `controller/shunt_banks.py`: `ShuntBank.available_directions(t_now,
  opposite_engaged=False)` drops the engaging direction (+1) when
  `opposite_engaged` is true; releasing stays allowed, so a change-over takes a
  release and, later, an engagement. `commit_selected(direction, t_now,
  opposite_engaged=False)` validates against the same set. New module function
  `opposite_engaged(bank, banks)`: true iff a bank of the other class with the
  same `dso_id` has a level > 0.
- `configs/config.py`: `shunt_sts_mutual_exclusion: bool = True`. Runs before
  2026-10-22 correspond to `False`.
- `experiments/runners/multi_tso_dso.py`: the per-instant bank bounds pass
  `opposite_engaged` over all banks; the commit loop evaluates it again at
  commit time, after the earlier commits of the same instant.
- `experiments/shunt_dispatch_check.py`: `--no-mutual-exclusion`; docstring
  updated.
- `tests/test_shunt_banks.py`: three tests (engaging direction removed only,
  helper requires same STS + other class + level > 0, commit rejects an
  engagement against an engaged opposite but allows the release).
- Thesis Ch 4 `ch:ofo:shunt`: the definition of $\mathcal{D}_b(k)$ now states
  the exclusion.

Gap, by design: the bound is per bank. If an MSC and an MSR of one STS are both
at level 0 and the MIQP engages both in the same solve, the bounds allow it; the
commit-time check then raises `ValueError` (fail-fast) instead of letting the
pair stand. Such a joint step costs two movement weights for nearly cancelling
effects, so it is not expected; it did not occur in the runs below. A strict
formulation would need a pairwise constraint in the MIQP.

## Verification

- `tests/test_shunt_banks.py tests/test_controller.py tests/test_miqp_solver.py
  tests/test_shunt_banks_integration.py`: 102 passed, 1 skipped. Full suite:
  see the end of this entry.
- Same 6 h AC check as in the previous entry (`shunt_dispatch_check.py`,
  paramset `ch9_selected_pcc27_gqceil`, W1 from 2016-01-05 08:00, +150 Mvar at
  bus 11 at min 10 / −150 Mvar at min 30, `g_w_tso_shunt = 10`, hold 1800 s),
  from a local `git archive` copy of HEAD `fa94411` plus the working-tree
  changes. Stored in `results/2026-10-22_sts_mutual_exclusion/` (`exclusion/`,
  `no_exclusion_repro/`, `banks_off/`, `comparison.json`, scripts).
- Reproduction: `--no-mutual-exclusion` reproduces the earlier unrestricted
  run exactly (same 8 commits, identical metrics), and `off` reproduces `off`.

| | off | restricted (HEAD) | unrestricted | unrestricted + exclusion |
|---|---|---|---|---|
| bank commits / releases | 0 / 0 | 3 / 0 | 8 / 0 | 4 / 0 |
| commits by class | – | 2 MSC, 1 MSR | 5 MSC, 3 MSR | 4 MSC |
| instants with > 1 bank per zone | 0 | 0 | 2 | 1 |
| STSs with MSC and MSR engaged | 0 | 0 | 3 | 0 |
| records outside 0.90–1.10 pu (TS / STS) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| TS V max / STS V max [pu] | 1.0502 / 1.0561 | 1.0502 / 1.0615 | 1.0507 / 1.0615 | 1.0507 / 1.0615 |
| mean TS V RMS error [pu] | 0.004778 | 0.004697 | 0.004580 | 0.004685 |
| interface-Q RMSE / abs. p95 [Mvar] | 0.581 / 0.517 | 0.572 / 0.492 | 0.640 / 0.556 | 0.643 / 0.548 |
| aggregate DER utilisation p95 | 0.282 | 0.289 | 0.277 | 0.348 |

## Findings

- **The exclusion works as intended.** At 720 s zone 2 engages the MSCs of
  DSO_1–3 together (as without exclusion); at 1800 s, after the load trip, no
  MSR is engaged. No STS carries an MSC and an MSR at the same time.
- **The engaged MSCs are never released.** Their holding time ends at 2520 s,
  but over the 107 TS instants from then on the per-level gradient entries of
  the three engaged zone-2 MSCs stay at most 1.42, 0.91 and 1.27 (median
  ≈ 0.07–0.29), far below the release threshold `g_w_tso_shunt = 10`
  (a release lowers the step objective iff c_b > g_w). The continuous
  actuators absorb the 75 Mvar instead, which is why aggregate DER utilisation
  p95 rises from 0.277 to 0.348 (+26 %; +24 % against banks off). The DSO_4 MSC
  second level at 1980 s of the unrestricted run does not occur here.
- **Net effect on tracking.** Interface-Q RMSE is unchanged against the
  unrestricted law (0.643 vs 0.640 Mvar; p95 0.548 vs 0.556), +12 % against
  the restricted law and +11 % against banks off. The TS voltage RMS error lies between
  the two (0.004685).

## Open points

- The exclusion removes the opposing pairs, but the underlying mechanism
  remains: the step cost prices movement only, and once the continuous
  actuators have absorbed a bank's surplus its gradient entry stays far below
  the movement weight, so an engaged bank stays engaged and binds DER reserve.
  Candidate remedies (design decision, not tuned here): a cost on the standing
  level or on the continuous reserve it binds, a smaller weight for the
  releasing direction, or a reserve-based release trigger.
- One scenario, one window; the voltage band was never close to binding.
- The joint engagement of both opposite devices from level 0 is detected (fail
  fast), not prevented, see "Gap" above.
