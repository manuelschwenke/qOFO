# SBX-H controller-intent terminal schedules

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** The active adapter used realised terminal-voltage snapshots as both
the settlement baseline and new controller references. With unit tracking
weight this still changed the autonomous 1.03 pu objective and made base SBX-H
physically different from `none` for no explicit support reason.

## Architectural decision

The default bilateral voltage schedule is now taken from each TSO
controller's intended terminal-bus setpoints. An explicit planning schedule may
override that intent. Measured plant voltages are not copied into controller
references.

Thus, for uniform 1.03 pu intent and `w_track_factor = 1`,

$$
J_V^{\mathrm{SBX,base}} = J_V^{\mathrm{none}},
$$

apart from numerical execution details. Base SBX-H supplies communication,
metering, hold/sag classification and remuneration. A planned-support interval
is the explicit physical intervention and changes selected terminal references.

## Implementation

- `SBXRunnerAdapter` constructs a constant schedule from the owning TSO
  controller's `v_setpoints_pu` at each corridor terminal.
- Planning JSON schedules retain precedence when configured.
- `build_default_contract` retains its snapshot fallback only for explicit
  low-level analytical callers; the active runner does not use it.
- `MultiTSOConfig.sbx_warmup_s` now defaults to zero because a controller-intent
  schedule is available before the first TSO tick.
- Experiment 015 now starts base SBX-H at minute zero and tests that its physical
  effect relative to `none` is approximately zero.
- The runner reports the schedule source and an initial terminal hold-margin
  pre-check. This diagnostic never rewrites the communicated promise.

## Assumptions, constraints and controlled outputs

- Intended setpoints must be finite, positive and aligned with monitored TSO
  voltage buses; corridor terminals must be uniquely monitored.
- Controlled outputs remain nodal TSO voltages and scheduled corridor-terminal
  voltages. Existing Q actuators, OLTCs, shunts, MIQP constraints and cached
  sensitivities are unchanged.
- Initial hold-margin is only a feasibility warning at one operating point, not
  a capability certificate.

## Open point

If a terminal cannot hold its intended schedule within
`v_hold_tolerance_pu`, contract prequalification must eventually either reject
the promise or agree another controller-aware schedule. The present
implementation reports this condition but deliberately does not redefine the
schedule from the measurement.

## Verification

A paired six-minute D0 run compared `none` and base SBX-H over 18 plant
steps. Every communicated terminal voltage was 1.03 pu. The maximum
difference between the two arms was:

- monitored zone voltage statistics: 0.0 pu;
- TSO OLTC tap positions: 0.0 tap steps.

A separate time-zero smoke run reported 3/10 terminals initially inside the
2.5 mpu hold tolerance and a worst margin of -29.47 mpu. This is now exposed
as a controller/contract feasibility result rather than hidden by lowering the
schedule.
