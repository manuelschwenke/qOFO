# Shunt refinement and consequences for the tuning campaign

Date: 20 October 2026. Status: opt-in experimental implementation.

The revised dispatcher requires persistent demand, can assist STS interface-Q
tracking, and can restore reserve near capability limits. Existing configurations
retain the original rule. The selected parameter sets and study defaults are
unchanged. The experimental gains below are not a new tuned parameter set.

## Control law and assumptions

The TS layer retains its MIQP and its continuous and discrete actuators. The
additional dispatcher operates the existing tertiary MSC/MSR banks. It observes
interface reactive powers, received STS references, reported capability
increments and measured voltages. It uses cached sensitivity models. It does not
read the plant model to make a control decision.

The STS controllers still track interface Q using their DER and OLTCs. The
normal TS output objective remains available to drive shunts. Tracking assistance
and reserve restoration are separate supervisory costs. A thesis statement that
*every* shunt request uses only the shared MIQP objective would therefore need
revision if this mode is adopted.

### Persistent demand

With positive `shunt_int_t_persistence_s`, a bank must observe demand of the same
sign and for the same reason throughout the confirmation period. Loss of demand,
direction reversal, a change of reason or a sample gap above 1.5 TS periods
clears the pending accumulation and restarts confirmation. Each committed step
requires a fresh confirmation period.

The existing gradient integration, half-step threshold plus hysteresis, bank
limits, dwell time and switching budget remain. Persistence and accumulated
gradient must both be sufficient. A large single sample cannot bypass
confirmation. A request that disappears during dwell cannot switch later.
Persistence means persistence at the sampled TS instants, not continuous-time
observation between samples.

Confirmed mode permits at most one bank commit per TS zone and instant, in the
existing bank order. The voltage guard includes the simultaneous local MIQP
prediction. It remains a local linear check, not a proof of feasibility of the
combined nonlinear, multi-zone response.

### Assistance for STS tracking

For the vector of STS interfaces, let `e = q_requested - q_measured`, and let
`D_d(e) = sign(e) max(abs(e)-d, 0)` act componentwise. The tracking cost is

    J_tracking = w_tracking ||D_d(e)||² / q_scale².

For a bank sensitivity vector `h`, its gradient is

    g_tracking = -2 w_tracking D_d(e)^T h / q_scale².

Feedback must be fresh and the received reference must have been present long
enough for an STS response. A full physical bank step must reduce the cost.
The same direction must still help the current TS command if it changed after
the feedback sample. Thus an obsolete STS error cannot alone trigger a switch.
Assistance can act inside the reported capability interval because an internal
network constraint or the STS objective may still prevent tracking there.

### Reserve restoration near the limits

Reserve support is considered only when tracking errors are within tolerance.
The raw capability increments are added to the corresponding measured interface
Q to obtain the absolute current band. Let its centre and half-width be `m`
and `b`. With `r = (q_requested-m)/b`, use

    J_reserve = w_reserve ||D_r0(r)||²,    r0 = 0.8.

The default leaves a margin of 10% of the full band at either limit. It exerts no
centring pressure inside the central band. Collapsed capability bands do not
produce a reserve request. A full step must reduce the cost and keep the
predicted reduced delegated target within the reported interval.

The capability reports are DER-based envelopes. They are not the full feasible
set under all voltage, current and tap constraints. Predicted reserve relief
assumes that the STS can subsequently track its reduced delegated contribution.

### Reference allocation at a switch

A normal objective-driven commit retains the feedforward convention. In support
mode its predicted effect is propagated to every coupled STS interface.

For an assistance or reserve commit, the same predicted vector is also
subtracted from the delegated TS control variable:

    shunt_offset_new      = shunt_offset_old + delta_q
    delegated_request_new = delegated_request_old - delta_q
    total_STS_reference   = delegated_request + shunt_offset.

The total reference therefore stays fixed at the switching instant. Increasing
only the offset would move the reference with the bank and preserve the
previous tracking error. The controller state, offset and cached shunt models
are updated in the same TS dispatch step.

## Verification

The focused regression suite passed: **73 tests**, 21.78 s. It covers the
existing integrator, output gradients and voltage rescaling, plus the new demand,
tracking, reserve, reference-transfer and configuration checks.

The reduced closed-loop tests verify two distinct behaviours:

- An unreachable 40 Mvar reference with a continuous actuator limited to
  ±20 Mvar is met after one 25 Mvar bank step. The continuous contribution falls
  to 15 Mvar.
- A tracked 45 Mvar reference with a continuous range of ±50 Mvar causes one
  reserve step. The continuous contribution falls to 20 Mvar and switching
  stops inside the central band.

Two AC power-flow tests hold continuous actuators fixed and construct an
interface request in each bank's effective direction. Both the MSC and MSR
reduce the actual interface error after confirmation. These are directional
plant checks, not evidence of an observed STS failure or a tuned closed-loop
reserve strategy.

Reproduce the tests with the project's `qOFO_clean` Python:

```text
python -B -X utf8 -m pytest -q -p no:cacheprovider
  tests/test_shunt_integrator.py tests/test_shunt_support.py
  tests/test_shunt_support_ac.py tests/test_shunt_integrator_integration.py
  tests/test_tso_output_gradient.py tests/test_shunt_voltage_rescaling.py
```

### One-hour AC comparisons

The first comparison uses the runner's current parameter set
`ch9_selected_pcc27_gqceil_comp_q_off`, the symmetric benchmark, real profiles,
minimum measurement noise with seed 20260717, TS/STS periods of 180/20 s,
25 Mvar bank steps and 30-minute dwell. Horizontal coordination is disabled
to isolate the vertical controllers. A 150 Mvar TS load is connected at
minute 10 and removed at minute 30. The objective gradients, actual selected
gradients, request reasons and commits are recorded separately.

Trial settings: confirmation 360 s, tracking weight 10, reserve weight 100,
tracking deadband 5 Mvar, normalisation 25 Mvar and response wait 40 s.
The original shunt movement weight of this parameter set is 0.06.

| Variant | Shunt commits | Interface Q RMSE [Mvar] | Absolute error p95 [Mvar] |
|---|---:|---:|---:|
| Original | 0 | 4.36584 | 11.72540 |
| Persistence only | 1 objective | 4.37093 | 11.66875 |
| Plus tracking assistance | 1 objective | 4.37069 | 11.63927 |
| Plus near-limit reserve | 1 objective | 4.37069 | 11.63927 |

Each run records 720 optimal STS solves. TS voltages span
0.97887–1.05006 pu and STS voltages 0.98435–1.05462 pu. These results show no
material tracking improvement. The assistance variants observe seven
consecutive tracking requests from minute 42 to 60, but their accumulated
gradient is insufficient to reach the switching threshold by the end.
No reserve request is selected in this scenario.

A second pair uses `ch9_selected_10_2026`, including its shunt movement
weight 0.1604992608, and the October campaign's doubled DSO_3 DER and active
load. The remaining short-run conditions are the same. The original and full
revised controllers produce identical reported metrics: interface Q RMSE
1.49634 Mvar, absolute error p95 1.44574 Mvar and no shunt commits. No support
request is selected. Each run records 720 optimal STS solves. This confirms
inactivity in this particular scenario. It does not establish equivalence on
the six 12-hour campaign windows or other weight candidates.

The extra objective switch in confirmed mode occurs at minute 45. Clearing
previous opposing accumulation can allow a later sustained request to reach
its threshold sooner. Adding a confirmation period therefore does not
guarantee fewer switches than the original integrator.

These short-run metrics are diagnostic and are not the campaign's complete
ranking metrics. A failed preliminary run with profiles disabled failed during
initial power flow in both variants, before shunt dispatch. The reported runs
retain the configured profiles.

Full configs, source hashes, logs, records and summaries are saved with each
local result under `tmp/shunt_refinement_20261019/results/`. The experiment
entry point is `experiments/shunt_refinement_check.py`. For example:

```text
python -B -X utf8 experiments/shunt_refinement_check.py
  --cases legacy confirmed assistance reserve --seconds 3600
  --output results/shunt_refinement
```

For the October selection comparison, add
`--paramset ch9_selected_10_2026 --frozen-network`. The completed pair is
stored in `results/october_selected` within the local review folder. All six
one-hour runs completed without an error, recovery or infeasibility entry in
their captured logs.

## Does the tuning campaign need to be repeated?

**Yes, its empirical part does if the revised dispatcher becomes the thesis
controller. No rerun is required merely because an inactive experimental option
has been added.**

This conclusion uses `docs/tuning/2026-10-16_protocol_experiment_B_rerun.md`,
including Amendments 1–3 and the result entered on 20 October at 08:30.
The current selection is `df17c0c32904`, stored as
`configs/paramsets/ch9_selected_10_2026.json`, with
`shunt_int_g_w = 0.16049926078643714`. The study defaults have not yet been
switched to that file.

| Campaign component | Consequence of adopting the new dispatcher |
|---|---|
| Frozen network and operating-point sensitivities | Reuse after checking identical network, operating points, bank positions and model construction. |
| Stage-I column scaling and spectral calculations | Reuse under those conditions. The MIQP has not changed. They do not establish stability of the added switching logic. |
| Sobol candidate vectors and analytical screen | Reuse if the Stage-I point, bounds and cached models remain identical. |
| Shunt calibration D2/D2b | Repeat or extend for the actual selected drive, confirmation resets and separate support costs. Freeze these settings before comparing candidates. |
| Stage-II closed-loop evaluation and selection | Repeat with the new dispatcher, including a new baseline. The previous ranking is not a ranking of this controller. |
| Isolated STS settling, Experiment C | Existing results can be reused for unchanged weights only after confirming identical initial states and held bank positions. Repeat for newly selected weights or changed initialisation. |
| Later thesis studies | Repeat affected runs with active shunts and any runs whose selected weights change. Unaffected fixed-actuator cases can be retained after an identity check. |

The latest protocol evaluates **all 118 analytical survivors plus the baseline
on W1–W6 for 12 hours**, which is **714 candidate-window simulations**.
Amendment 2 removed the short-run shortlist. W5 is executed and reported as
stress evidence but is excluded from ranking. Re-evaluating only the previous
winner would validate that candidate, not reproduce the declared selection.
The baseline must use the same revised controller because the admissibility
filter includes an error bound relative to that baseline. The settling filter
and the recomputed 2% ranking band of Amendment 3 still apply.

The previous shunt calibration maps a p90 normal-objective gradient to a nominal
60-minute accumulation time. It does not account for resets, minimum
confirmation, changing request reasons, full-step support checks or support
weights. The recorded `zone_tso_shunt_g_h` still denotes the original objective
gradient. Calibration of this variant must also inspect
`zone_tso_shunt_drive` and `zone_tso_shunt_request_kind`.

Use a new result directory. The current Stage-II worker reuses a result when
its output file exists, and the weight fingerprint alone does not identify
the controller implementation or these new settings. Freeze and record the
complete controller configuration and source snapshot. Before a production
campaign, extend the parameter provenance to include the support weights,
deadbands and timing. If the common objective scale changes, the support cost
weights and gradient deadband must scale consistently with the shunt movement
weight. The current tuning parameter registry does not yet carry these new
fields.

## Recommendation and remaining limits

Keep the completed October campaign as the record of the original controller.
Keep this dispatcher optional while checking targeted persistent-error and
near-limit scenarios over enough time to exercise support commits. The
short comparisons do not justify replacing the selected settings. After the
shunt design and its parameters are fixed, repeat the affected empirical
selection once.

The implementation supports the cascaded OFO TS/STS path. Central control,
local-controller variants, partial STS selections and vertical SBX settlement
are rejected when support is enabled. Horizontal coordination remains
available but was disabled in these diagnostic runs. RMS co-simulation and
multi-day switching behaviour have not been validated here.

The per-bank voltage guard can reject a beneficial partial correction while
another observed voltage remains outside its corridor. Bank arbitration uses
fixed order, not a global optimisation. Existing daily budgets use reset
windows, not a true rolling 24-hour counter. These limitations are unchanged
and should not be mistaken for guarantees supplied by the new logic.
