# 2026-07-13 — Repository cleanup continuation and 014 SBX-H repair

## Context and reason

Continued the approved strict repository cleanup from recovery commit `af50767`.
During the cleanup, the legacy ΔV-reference coordination record block was removed
too broadly. The physical per-tie reactive-flow field `tie_q_mvar` was removed
with coordination-only fields even though
`experiments/runners/_multi_tso_helpers.py::_record_zone_live_plot_observables`
still populates it. This caused archived experiment 014 to fail after the first
power-flow record was assembled.

## Changes made

- Restored `MultiTSOIterationRecord.tie_q_mvar: Dict[int, float]` as an
  explicitly monitoring-only field. It is not an objective, constraint,
  setpoint, or horizontal coordination signal.
- Corrected `experiments/archived/014_SBX_SINGLE_DEMO.py` after its move:
  - repository root now uses `Path(__file__).resolve().parents[2]`;
  - removed stale BME and ΔV-reference configuration attributes;
  - removed the unrelated contingency list that overwrote the demonstrator's
    intended stress event list;
  - retained `coordination_mode="sbx_h"` and the SBX-H configuration.
- Continued the approved cleanup:
  - removed BME controller/runner seams and legacy Q-tie objective/slack/band;
  - retained Q-tie measurement and sensitivity rows;
  - removed the ΔV-reference coordinator path;
  - renamed `configs/multi_tso_config.py` to `configs/config.py` and updated
    imports;
  - pruned removed-mechanism configuration fields;
  - restored the CIGRE V1–V5 variant ladder;
  - moved historical experiments into `experiments/archived/`.

## Assumptions and model boundary

- Controllers continue to act only on measurements and cached sensitivities;
  no controller receives direct plant-model access.
- Physical tie-line Q remains an observed/recorded controlled-output diagnostic.
  It no longer contributes an objective term or soft constraint.
- Experiment 014 uses the existing TSO/DSO actuators from the CIGRE base
  configuration and controls zonal/interface voltages and reactive-power
  exchange through SBX-H scheduled boundary support.

## Verification

Executed a headless 014 smoke run with:

- horizon: 2 minutes (6 simulation steps at `dt_s=20 s`);
- SBX start: 0 minutes;
- reactive stress: +50 Mvar at minute 1;
- all live plots disabled for the smoke run.

Result: `014 smoke OK 6 steps tie_samples 30`.

## Cleanup completion update — 2026-07-13 16:45 CEST

- Added `experiments/results_io.py` with numbered timestamped run roots,
  exact `config.pkl`, readable `config.json`, `meta.json`, standard
  subdirectories, and latest-run lookup.
- Wired result roots into the manual OFO runner, SBX-H comparison,
  SBX-V demonstrator, and CIGRE 005/006. CIGRE 006 propagates its selected
  root to Windows worker processes for parallel Monte-Carlo runs.
- Added `visualisation/plot_sbxv.py`. The observational live figure reads
  adapter-side normal bands, 15-minute meters, request/grant logs, and
  settlement inputs; it does not feed a controller.
- Added `experiments/demonstrate_sbxv.py` with a fixed ±25 Mvar band and a
  reactive-load stress at transmission bus 9.
- Added explicit CIGRE/archive package markers and refreshed repository and
  experiment READMEs.
- Removed the final active-code references to BME, the ΔV-reference
  coordinator, `tso_g_q_tie`, the removed phasor-loss switch, and the
  retired Stage-2 tuning field.
- BO tuning is now explicitly eight-dimensional; `g_w_pcc` was restored to
  its documented upper cap of 30.

## Verification update

- Compile/import health passed for all retained packages and experiment
  entry points.
- Results helper smoke passed in a temporary directory.
- SBX-V plotter passed a headless band/meter/scheduler/settlement smoke.
- A real two-minute SBX-V runner smoke passed with 6 records and 4
  AggregationAreas.
- Restored CIGRE V1, V4, and V5 each passed a two-minute, six-record
  runner spot-check.
- Pytest initial result: 593 passed, 11 skipped, 10 failed. The ten failures
  reduced to two cleanup causes and were corrected:
  archived 014 import path and stale `g_w_dso_der_vref` tuning override.
- Focused regression rerun: 15 passed.
- The complete tuning suite and two long runner tests reached their 10-minute
  and 6-minute verification caps, respectively, after progressing beyond
  the former immediate config-construction failure; no failure output was
  produced before timeout.
- Strict active-code dangling-reference grep is clean.

## Open points

- The long end-to-end tuning simulations should be rerun without a wall-clock
  cap when compute time is available.
- Full-horizon CIGRE V1/V4/V5 and 30-minute SBX-H comparisons remain
  computational validation runs; their short runner spot-checks and
  entry-point imports pass.
