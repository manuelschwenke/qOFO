# 2026-07-13 - Minimal SBX-H hold/sag settlement

Timestamp: 2026-07-13 11:21:42 +02:00
Scope: active sbx_h v6 path only
Reason: simplify the current v6 attributed-deviation mechanism to the
agreed scheduled-voltage-reference rule. The archived v5 implementation
under _archive/sbx_h_v5 was intentionally left unchanged.

## Architectural decision

SBX-H remains a planned boundary-voltage coordination mechanism. Each
TSO tracks the bilateral terminal-voltage schedule with its own local
controller. Runtime capability offers, requests, matching, grants, and
strength products are not part of the active mechanism.

Settlement now remunerates delivered reactive support energy only when:

1. exactly one corridor side sags relative to its active voltage
   schedule;
2. the opposite side remains inside its absolute holding tolerance; and
3. the beyond-band reactive-flow deviation points from the holding side
   toward the sagging side.

A side being merely better than its neighbour is insufficient: it must
satisfy its own absolute scheduled-voltage holding criterion.

## Method

For every corridor settlement window, using the corridor orientation
positive Q from area A to area B:

- e_A = min_l(V_A,l,meas - V_A,l,sched)
- e_B = min_l(V_B,l,meas - V_B,l,sched)
- side X holds if e_X >= -epsilon_hold
- side X sags if e_X < -epsilon_sag
- epsilon_hold must be smaller than epsilon_sag

The reference flow is recomputed with the contracted pi-line model at
the active scheduled terminal voltages and measured active transfer:

Q_0 = sum_l q_l(V_A,l,sched, V_B,l,sched, P_l,meas)

and

Delta Q = Q_meas - Q_0.

Paid support above the deadband B_Q is:

- A sags and B holds: Q_sup = max(0, -Delta Q - B_Q)
- B sags and A holds: Q_sup = max(0,  Delta Q - B_Q)
- otherwise: Q_sup = 0

An optional Q_sup cap limits commercial exposure. Payment is

C_sup = p_sup * Q_sup * T_window,

with the sagging side as payer and the holding side as payee. Per-window
bilateral payment conservation is asserted.

The measured-P baseline deliberately removes ordinary P-transfer changes
from the paid Q-support quantity. No first-order C_A/C_B/C_P attribution,
dominant-causer penalty, UNATTRIBUTED state, or in-band netting ledger
remains.

## Assumptions

- Both parties share the corridor orientation, line parameters, active
  voltage schedule, deadband, price, and role thresholds.
- Terminal P, Q, and voltage measurements are time-aligned and averaged
  over the same settlement window.
- Positive corridor Q is export from area A to area B.
- The pi-line baseline is sufficiently accurate for settlement after
  calibration and metering validation.
- The first scheduler boundary remains an initialization boundary; the
  next elapsed window is the first settled window.

## Constraints, actuators, and controlled outputs

Constraints:

- local nodal voltage and branch/current constraints remain enforced by
  each TSO controller;
- hold and sag use absolute schedule-relative thresholds;
- settlement requires the correct aggregate corridor-Q direction;
- persistent beyond-band flow and area voltage violations still feed
  only the slow A4 re-planning escalation.

Actuators:

- unchanged local TSO actuators: AVR voltage references, OLTCs,
  MSC/MSR, and TS-connected DER;
- underlying HV controllers continue tracking their Q assignments with
  local actuators;
- SBX-H itself applies only the agreed boundary-voltage schedule and
  does not dispatch a bilateral deal at runtime.

Controlled outputs:

- scheduled corridor-terminal voltage magnitudes;
- corridor reactive-power flow/deviation for metering and escalation;
- local nodal voltages and equipment constraints through the existing
  zone controllers.

The controllers remain plant-agnostic and operate through their cached
models/sensitivities and the measurements supplied by the runner.

## Strength treatment

No strength product is sold. An optional ex-post diagnostic reports

S_obs = Q_sup / max(0, -e_holder)

in Mvar per mpu when the holder is slightly below schedule. It has no
effect on eligibility, quantity, or price and is not a Thevenin-strength
estimate.

## Files changed

- sbx_h/config.py: replaced deviation-penalty and attribution knobs with
  support price, optional cap, and hold/sag thresholds.
- sbx_h/contract.py: froze the new bilateral settlement terms.
- sbx_h/settlement.py: replaced causal attribution with the minimal
  schedule-relative directional support-energy rule.
- sbx_h/scheduler.py: supplies measured P and active scheduled voltages
  to settlement and records support roles/payments.
- sbx_h/adapter.py and sbx_h/__init__.py: updated active-path
  documentation.
- tests/sbx_h: replaced attribution tests with hold/sag, direction,
  P-baseline, cap, rolling-window, conservation, and scheduler
  integration tests.

## Verification

- Python compilation succeeded for the modified active modules and
  focused tests.
- Focused SBX-H suite: 41 passed in 10.45 s.

## Open points

- Calibrate epsilon_hold, epsilon_sag, B_Q, and p_sup from measurement
  error, natural corridor variability, and operational value.
- Decide whether multi-line corridors need per-line eligibility instead
  of aggregate Q settlement.
- Define treatment of data gaps, bad data, schedule discontinuities, and
  disagreement over meter values.
- Validate whether the measured-P pi-line baseline is accurate enough
  for financial use; otherwise keep this as a simulation settlement
  metric only.
