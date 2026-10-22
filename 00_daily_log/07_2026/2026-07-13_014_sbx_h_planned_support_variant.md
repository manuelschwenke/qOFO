# Experiment 014 planned neighbour-support variant

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** Manuel requested an SBX-H single-demo variant in which the area-3
load connection is known in advance and neighbouring areas are therefore
asked to raise their corridor-terminal voltage schedules during the event.

## Code changes

- `experiments/archived/014_SBX_SINGLE_DEMO.py`
  - preserves the existing behaviour as the default `base` variant;
  - adds `planned_support`, selectable with `--variant planned_support`;
  - adds `--support-dv-mpu`, defaulting to `+2.5 mpu`;
  - restores the documented 300-minute default horizon;
  - schedules the increase at the area-1 and area-2 sides of corridors
    `(1,3)` and `(2,3)` from load connection until load disconnection (or the
    simulation horizon, whichever comes first);
  - records the selected variant and planned-support request in the experiment
    report and uses a variant-specific default result directory.
- `tests/sbx_h/test_demo_config.py`
  - verifies that `base` has no support intervals;
  - verifies the corridor sides, voltage increment and exact event window for
    `planned_support`.

## Method and interpretation

The variant uses the existing SBX-H v6 planned-support schedule product. For
sorted corridor keys `(1,3)` and `(2,3)`, areas 1 and 2 are side A, so the
configured interval is `(t_on, t_off, +dv_A, 0)`. This raises each supporting
area's own terminal reference. The SBX-H controller tracks that scheduled
promise and the existing settlement mechanism evaluates performance against
it. No run-time deal negotiation or plant foresight is introduced into the
controller.

## Assumptions and limits

- The load-connection timing and requested voltage increment are exogenous
  planning information.
- `+2.5 mpu` is a configurable experiment value, not an optimized magnitude.
- The existing controller settings, including `w_track_factor=2.0`, remain
  unchanged.
- Schedule transitions are enacted at the existing SBX/TSO sampling times;
  the default event times are aligned with those cycles.

## Verification

- Python compilation passed for the experiment and focused test module.
- Focused configuration tests: **2 passed**.
- Complete SBX-H test suite: **51 passed**.
- An 8-minute headless `planned_support` smoke run completed 24 plant steps
  and wrote the figure, settlement outputs and experiment report.
