# 2026-07-13 - SBX-H 300-minute live capability experiment

Timestamp: 2026-07-13 11:45:38 +02:00
Scope: experiment 014, reusable SBX-H live visualization, and explicit
scheduler telemetry for hold/sag state.

## Reason

Create one clean thesis-oriented experiment that demonstrates the active
SBX-H v6 mechanism over a 300-minute horizon. The previous experiment
014 depended on the multi-arm comparison experiment 015 and its live
plot still described the removed attribution/netting/deal mechanism.

The revised experiment is self-contained apart from the established
005 network/configuration factory and the common multi-TSO/DSO runner.

## Scenario

Default no-argument run:

- horizon: 300 min;
- plant/controller warm-up: minutes 0-30;
- SBX-H contracts frozen: minute 30;
- reactive sink connected: 900 Mvar at bus 15, minute 60;
- reactive sink removed: minute 240;
- recovery observation: minutes 240-300;
- TSO period: 180 s;
- SBX-H settlement window: k_sched=2, hence 6 min.

The old tie-gradient coordination is disabled so only SBX-H steers the
corridor-terminal voltage references.

## Visualized signals

For each corridor, the live thesis figure displays:

- measured reference-end corridor Q_meas;
- Q_0 evaluated from scheduled terminal voltages and measured P;
- the Q_0 +/- B_Q deadband;
- signed paid Q_sup;
- minimum measured terminal voltage on side A and side B;
- scheduled terminal-voltage references on both sides;
- per-window state strips for A and B:
  hold=True, sag=True, or neither;
- persistent-exceedance escalation markers.

The bottom row shows signed Q_sup for all corridors and cumulative
bilateral payments. A status line prints the latest Q_0, Q_sup, and
support state for every corridor.

## Assumptions

- Positive corridor Q and signed Q_sup point from area A to area B.
- Terminal P, Q, and voltage measurements are aligned with the active
  voltage schedule and the settlement window.
- The minimum terminal voltage shown per side is a visualization of the
  multi-line boundary; settlement eligibility still uses the worst
  schedule-relative terminal error.
- Strength remains diagnostic only and is deliberately not plotted as a
  traded product.
- Experiment 005 remains the canonical source of the tuned CIGRE/IEEE-39
  multi-zone plant and hierarchical-controller configuration.

## Constraints, actuators, and controlled outputs

Constraints:

- local nodal-voltage and branch/current constraints remain in each
  zone controller;
- hold and sag are absolute schedule-relative conditions;
- payment requires one sagging side, one holding side, and correctly
  directed beyond-band Q;
- the A4 signal is an escalation for slow re-planning, not an immediate
  bilateral dispatch.

Actuators:

- local AVR voltage references, OLTCs, MSC/MSR, and TS-connected DER;
- underlying HV controllers and their local actuators are unchanged;
- SBX-H writes scheduled terminal-voltage references only.

Controlled outputs:

- corridor-terminal voltage magnitudes;
- measured corridor reactive-power flow;
- local nodal voltages and equipment limits;
- settlement and escalation telemetry do not directly control the plant.

Controllers remain plant-agnostic and operate through cached
models/sensitivities and supplied measurements.

## Files changed

- experiments/014_SBX_SINGLE_DEMO.py:
  self-contained 300-minute experiment, CLI overrides, clean outputs,
  scenario summary, and headless/live use of the same plot path.
- visualisation/plot_sbx.py:
  replaced obsolete deal/attribution panels with Q_0, Q_sup, terminal
  voltages, hold/sag strips, and bilateral payments.
- sbx_h/scheduler.py:
  exposed a_sags, b_sags, a_holds, and b_holds in each corridor cycle
  record for visualization and evaluation.
- configs/multi_tso_config.py and
  experiments/runners/multi_tso_dso.py:
  updated stale active-path documentation.
- tests/sbx_h/test_scheduler.py:
  verifies the explicit scheduler state booleans.

## Verification

- Python compilation succeeded for the experiment, plotter, scheduler,
  configuration, and runner.
- Focused SBX-H suite: 41 passed in 11.03 s.
- 75-minute headless smoke run:
  - 225 plant steps;
  - 176.3 s wall time;
  - all TSO solves accepted;
  - final PNG, ledger, settlement summary, and experiment report written;
  - one paid window;
  - paid support: 34.849 Mvar for 0.1 h = 3.485 Mvar h;
  - gross payment: 17.42 EUR from area 2 to area 1.

The smoke run exercised the exact live-plot update path on the Agg
backend. The full 300-minute default was not executed during this
implementation turn; at the observed smoke speed it is expected to take
roughly 12 minutes, subject to solver variability.

## Observed mechanism behavior and open points

The deep area-3 stress initially made both sides sag on corridors (1,3)
and (2,3). Those windows were correctly not paid. On corridor (1,2),
area 1 subsequently held while area 2 sagged and Q flowed from area 1 to
area 2, so the window was remunerated.

This is an academically relevant system response, but it means the
default event demonstrates network-wide redistribution rather than a
simple direct supporter-to-area-3 story. The full 300-minute run should
be inspected before selecting the final thesis figure. If a more
pedagogical direct-corridor example is required, the sink magnitude
should be calibrated in a separate parameter study rather than chosen
to force a desired settlement outcome.

Further open points:

- verify plot readability on the target thesis page size;
- calibrate the event magnitude and hold/sag thresholds;
- decide whether all corridors or only corridors incident to the
  stressed area should appear in the final thesis panel;
- archive or remove experiment 015 only after its comparison evidence
  has been retained elsewhere.
