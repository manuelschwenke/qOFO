# 2026-07-14 — SBX-H automatic restoration experiment

## Timestamp

2026-07-14, Europe/Berlin.

## Reason

Test whether a minimal spontaneous support request can accelerate voltage
restoration after an unforeseen terminal-voltage violation without changing the
permanent SBX-H contract reference. Define objective gates and retain the simple
settlement-only architecture as the explicit rollback.

## Changes

- Added opt-in `support_control_mode="automatic"`; the default remains
  `settlement_only`.
- Added a deterministic per-route trigger, escalation, hysteretic release,
  gradual withdrawal, failed-withdrawal restoration, and security/direction
  abort state machine.
- Kept contract baseline references separate from temporary supporter-side
  controller commands.
- Added exact-side incremental corridor-Q attribution, pre-episode baseline
  subtraction, `B_Q` deadband, cap, bilateral payment ledger, full state trace,
  and CSV/Markdown outputs.
- Suppressed legacy hold/violation payment in automatic mode to prevent double
  remuneration; its classifications remain available as diagnostics.
- Extended archived experiment 014 with `automatic_support` and an autonomous
  paired `--validate` mode.
- Extended the 4x2 live plot to distinguish contract baseline from runtime
  command and to show automatic state/level, attributed response, runtime paid
  support, and runtime payments.
- Added immutable validation gates and a machine-readable rollback decision.
- Added deterministic unit tests for activation, escalation, recovery,
  failed-withdrawal restoration, insecure abort, payment attribution, and gate
  evaluation.

## Method structure

The requesting area is observed against its configured security band. The
supporting area receives only a bounded voltage-reference bias at its own
corridor terminals. Settlement compares measured corridor Q with a
counterfactual that removes the supporter's voltage displacement while retaining
the requester's measured voltage and measured active transfer.

## Autonomous validation result

Paired experiment: 36 min, 108 plant steps per arm, 500 Mvar sink at bus 15
from minute 6 to 24, area 3 requests area 1, 2.5 mpu per level, maximum
level 2, 10 Mvar settlement deadband. The full 300 min script remains the
demonstration entry point; the shorter paired horizon was selected to include
pre-event, sustained stress, removal, and complete withdrawal at lower cost.

Final gate verdict: 7/8 passed; `G4_restoration_accelerated` failed. The
machine-readable decision is `rollback_to_settlement_only`.

- G1 PASS: pre-event voltage-stat difference 0.000e+00 pu; tap difference 0.
- G2 PASS: armed at 3.333 min; activated once at 9.333 min.
- G3 PASS: maximum command 5.00 mpu; zero contract reschedules.
- G4 FAIL: area-3 undervoltage exposure 0.431323605 -> 0.431223804
  pu min, only 0.023138% improvement versus the required >0.5%. Worst depth
  remained 36.232528 mpu; mean depth changed 23.844277 -> 23.838833 mpu.
- G5 PASS: all solves accepted; supporter voltage 0.996810 to 1.044898 pu,
  inside 0.90 to 1.10 pu; no security abort.
- G6 PASS: one activation, one escalation, one withdrawal, one closure, no
  restore/direction flip, final level 0.
- G7 PASS: maximum payment imbalance 0 EUR; all paid rows valid; no parallel
  legacy payment.
- G8 PASS: maximum attributed directional response 158.678198 Mvar.

Runtime settlement recorded four active windows and two paid windows,
29.542768 Mvar h, and 147.713840 EUR from area 3 to area 1. Both paid
windows occurred after the sink-removal time; the two active windows during
the stressed interval had negative attributed directional response and no
payment. This is additional evidence that the tested runtime command did not
provide timely voltage restoration, despite a traceable lagged Q response.

Tests after implementation: 57/57 SBX-H tests passed before the arming
refinement; targeted refined relay/gate tests passed 7/7 and 5/5. A final
complete SBX-H regression run is recorded below after documentation updates.

## Decision

Keep `settlement_only` as the default and reject automatic runtime support as
a working restoration mechanism for this scenario. Retain the implementation
only as an opt-in, reproducible failed experiment. The accepted fallback is
ex-post remuneration relative to permanent baseline physics without realtime
coordination, plus planning-phase higher terminal-voltage schedules for known
events.

## Final regression

- `python -m pytest -q tests/sbx_h`: 59 passed in 30.38 s.
- Experiment artefacts and gate report: `results/014_SBX_H_DEMO/validation_36min_final2/`.

## User-approved architectural rollback

After reviewing the failed restoration gate, the active automatic request, temporary command, attributed runtime settlement, validation module, and automatic/planned-support experiment variants were removed. The accepted active architecture is ex-post remuneration for useful physical Q support plus planning-phase voltage schedules only. For the current experiments, the planning schedule is explicitly constant at 1.03 pu for areas 1, 2, and 3; no schedule changes are injected. Contract schedule composition remains available as planning infrastructure outside experiment 014.
