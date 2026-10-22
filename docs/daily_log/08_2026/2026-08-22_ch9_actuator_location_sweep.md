# 2026-08-22 — Ch. 9.1 actuator-location settling sweep

**Timestamp:** 2026-08-22 01:43 CEST  
**Reason:** Redesign open-loop experiment A so the spatial choice of actuator
steps is defensible, currents and disturbances are excluded, and the search
for the worst settling time remains feasible as an overnight RMS campaign.

## Assumptions and scope

- Operating point: `full_t0_20160105-0800.json`, with every isolated RMS case
  advanced by 300 s from `ComInc` before its command is armed.
- Actuators: TSO and DSO Q resources (including the DSO-controller `WPC_*`
  columns), dispatchable synchronous-machine AVR references, all non-slack
  TSO OLTCs, all DSO coupling OLTCs, and all tertiary MSC/MSR banks.
- Slack/equivalent AVR and its machine transformer are excluded because the
  production controller withholds these actuators.
- Controlled outputs: TSO and DSO nodal voltages, all 12 DSO interface-Q
  flows, and the five inter-zone TSO tie-Q flows. Currents are not monitored.
- No disturbance event (load, outage, switch contingency, profile change) is
  created.
- Settling bands are fixed at 0.001 pu for voltage and 1 Mvar for interface Q;
  there is no relative 2% term. The final value is a 5 s mean and at least 5 s
  of in-band confirmation is required.

## Method implemented

Added
`experiments/ch_9_parameter_selection/ch_9_1_actuator_location_sweep.py`.

1. Build the feasible one-command catalogue from the snapshot actuator
   inventory and the same VDE-AR-N-4120-v2 Q capability rule as the controller.
2. Run both feasible directions at every discrete location in RMS:
   24 coupler-OLTC cases, 14 non-slack TSO-OLTC cases, and 8 shunt switch-in
   cases (46 total).
3. Rank 66 feasible continuous Q/AVR commands by the maximum normalized AC
   power-flow change over the controlled-output vector. Select the top two per
   actuator class/direction, the top location per zone/DSO, a deterministic
   median audit, and every failed QSS case. This selects 24 continuous RMS
   cases, for 70 RMS cases in total.
4. Run each case independently with a non-persistent event created after the
   pre-settle; event-admission barriers are executed before the event is
   crossed. An actuator-state diagnostic must prove the command fired.
5. Begin with a 30 s response horizon and rerun censored cases at 60 and at
   most 120 s. Write results, signal metrics, summary, failures and provenance
   after every completed case so the campaign is resumable.

## OLTC representation

The old TAPCTRL used a 5 s PT1 and therefore introduced a slow ramp rather
than a contact action. For this experiment only, an OLTC command is issued
after an explicit 5 s command-to-contact delay and the TAPCTRL time constant
is temporarily set to 0.05 s (five fixed RMS steps). Settling is measured from
the original dispatch time, so the mechanical delay remains in the reported
time. The original TAPCTRL `Tmech` is restored in `finally` after every tap
case. Shunts remain direct `EvtTap` actions.

## Verification

- `python -m py_compile .../ch_9_1_actuator_location_sweep.py`: passed.
- `pytest tests/experiments/test_ch9_actuator_location_sweep.py -q`:
  **6 passed** (final run: 12.69 s).
- Offline QSS dry run: 46 discrete + 66 continuous candidates; 70 scheduled;
  output vector = 70 V + 12 DSO-Q + 5 TSO-Q, zero currents/disturbances.
- Full live output preflight: passed after aligning the drift gate with the
  declared physical bands. Worst unforced 60 s range after the 300 s
  pre-settle was 0.981 Mvar at `Q_TSO_Z2_Z1_TN_line14` (98.1% of band).
- Direct MSC probe `shunt_p1_SH_MSC_DSO_1_s0`: event diagnostic passed;
  worst settling 0.022 s at `Q_DSO_NC3W_DSO_1_t0`.
- Fast-PT1 OLTC probe `tap_m1_NC3W_DSO_1_t0`: actual `c:n3tap_h` diagnostic
  passed; worst dispatch-to-settled time 5.912 s at
  `Q_TSO_Z2_Z1_TN_line14`.

Two earlier preflight folders intentionally record rejected provisional gates
(0.1 Mvar and 0.0001 pu). They contain no actuator result and must not be used
as experiment data.

## Overnight launch

- Started hidden at 2026-08-22 01:42:09 CEST.
- PID: `108040`.
- Result directory:
  `results/timescale/actuator_location_sweep_t0/20260822-014209`
- Internal log: `.../20260822-014209/run.log`
- Bootstrap stdout/stderr:
  `results/timescale/actuator_location_sweep_t0/bootstrap_20260822-014209.{out,err}.log`
- Initial verification: process alive, QSS catalogue complete, metadata and
  run log written, PowerFactory connected at fixed 10 ms RMS step, campaign
  preflight passed at the same 0.981 normalized worst drift, stderr empty, and
  case 1/70 (`tap_m1_NC3W_DSO_1_t0`) started.

## Risks and interpretation

- The discrete-location statement is exhaustive. The continuous-location
  statement is “worst RMS-verified after deterministic QSS screening,” not a
  mathematical maximum over every continuous location. The median audit is a
  check against a misleading amplitude ranking, but QSS amplitude cannot
  guarantee dynamic-mode ordering.
- The unforced tie-Q range is close to the 1 Mvar settling band (0.981 Mvar).
  It is recorded and cases with a drifting final tail are censored/extended,
  but near-band settling results should be interpreted with this baseline
  uncertainty.
- Repeating the 300 s pre-settle dominates runtime. At the measured wall-clock
  rate, 70 cases are expected to take approximately 12–16 h, longer if many
  cases need adaptive reruns. The run is intentionally resumable.

