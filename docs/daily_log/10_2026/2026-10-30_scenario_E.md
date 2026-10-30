# Scenario E: reactive deficit beyond area 2's own TS resources

Timestamp: 2026-10-30, ~21:00–24:00 Europe/Berlin.
Branch `weekend/scenario-E`, from `weekend/harmonised-limits-2026-10-30` at 8ebc032. Worktree
`F:\qOFO_scarcity`, kept separate so that the running STEP / Case 1 campaigns in `F:\qOFO_weekend`
stay untouched.
Session: side session of the Case-2 main session, at the author's direct instruction.

## Why

- The weekend REF runs gave OFO+ − OFO = −0.39 / −0.47 / −0.43 mpu (A / B / D).
- The author asked for a Q-scarce scenario that shows when access helps very much, and suggested
  the outage of 9–39.
- A plant-only bound showed that B and D were not scarce for OFO. A deficit of ≥ 600 Mvar at IEEE 4
  (with G3 and 9–39 out) exceeds area 2's own TS resources.
- Declaration: `docs/analysis/2026-10-30_scenario_E_declaration.md`.

## What changed

| file | change |
|---|---|
| `experiments/ch_11_case2_multisystem/ref/scenarios.py` | new entry `"E"`: `_gen(1, _at(30, 40))`, `_line(14, _at(120, 100))`, `_load(3, 600.0, _at(210, 140))`; module docstring names line 14 = 9–39 |
| `docs/analysis/2026-10-30_scenario_E_declaration.md` | new: schedule, plant criterion with measured bounds, hypotheses, limits |

No controller, plant or metric code changed. `--scenario E` works with the existing `run_ref`.

## Checks

- `sc.describe("E")`: 30:40 trip gen 1, 121:40 trip line 14, 212:20 connect 600 Mvar at bus 3. All
  events lie off the dispatch grid (phases 40 / 100 / 140 s).
- `assert_case2_controlled` passes.
- Runner set-up for L2 and O3:
  - line 14 is IEEE 9–39;
  - gen 1 is G3;
  - the dormant 600-Mvar load is created at bus 3.
- Not checked: closed-loop convergence after 212:20. The open-loop power flows of the end state
  converge; a failed variant is recorded as a result by `run_ref`.

## Run

`run_ref --scenario E --variants L2,S1,S2,O2,O3,M --jobs 6` on e5wimi, queued to start when Case 1
K1-ALL has ended (its 6 workers).
