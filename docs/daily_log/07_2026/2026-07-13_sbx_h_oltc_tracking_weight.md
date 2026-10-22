# SBX-H terminal tracking weight and OLTC activity

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** Investigate the observation that SBX-H can worsen the aggregate
three-area voltage-tracking error relative to `coordination_mode="none"`, with
increased TSO OLTC switching as the suspected cause.

## Assumptions and evaluation scope

- Plant/scenario: stored `015_SBX_COMPARE` D0 v6 case, 150 Mvar reactive sink
  connected at minute 60; equal 120-minute comparison horizon.
- Controlled outputs: monitored TSO-bus voltage error and corridor-terminal
  voltage references.
- Actuators: existing TSO generator/DER Q, OLTCs and shunts; no actuator or
  constraint was added or removed.
- Controllers use only measurements and their cached sensitivities; the test
  did not introduce plant-model access.
- Diagnostic aggregate: mean of the recorded per-zone voltage RMS errors after
  the initial six samples. OLTC travel is the sum of absolute tap increments.

## Established result

| Arm | terminal weight | aggregate RMS [mpu] | area-1 tap travel | area-1 reversals |
|---|---:|---:|---:|---:|
| none | ordinary | 7.782 | 0 | 0 |
| stored SBX-H | 20 x ordinary | 8.115 | 4 | 2 |
| diagnostic SBX-H | 1 x ordinary | 7.921 | 0 | 0 |

The 20-fold terminal weight was inherited from the removed v4/v5 runtime-deal
mechanism. With `g_v = 1e7`, it produced a terminal weight of `2e8`, while the
TSO OLTC movement regularisation in this experiment is `1e4`. At disturbance
onset the MIQP therefore selected machine-transformer taps to reduce the
high-priority terminal error and reversed them after recovery. Reducing the
factor to one removed all additional area-1 tap travel and reversals in D0.

## Code revision

- Changed `SBXConfig.w_track_factor` default from 20.0 to 1.0.
- Pinned `w_track_factor=1.0` in experiment 015 for reproducibility.
- Updated active v6 documentation and added a regression test for the neutral
  default.
- Added post-warm-up aggregate voltage RMS, TSO OLTC travel and TSO OLTC
  zone-event diagnostics to experiment 015's result matrix.
- Settlement, hold/sag thresholds, scheduled references, MIQP constraints,
  actuator availability and OLTC cooldowns remain unchanged.

## Residual and open question

The diagnostic still has a small +0.139 mpu aggregate gap to `none`, but no
additional area-1 OLTC movement. This residual is caused by comparing different
voltage objectives: `none` retains 1.03 pu at every monitored bus, whereas the
default SBX contract replaces terminal targets by feasible warm-up snapshots
(stored D0 values range down to 1.01192 pu). It is therefore not evidence of
remaining OLTC chattering.

A possible later architectural refinement is a one-sided contractual floor
penalty,

$$
J_{\mathrm{SBX},V}=w_{\mathrm{SBX}}
\sum_{i\in\mathcal T}\left[\max\left(0,
V^{\mathrm{sched}}_i-\varepsilon_{\mathrm{hold}}-V_i\right)\right]^2,
$$

added to (rather than replacing) the ordinary voltage objective. It would not
pull a terminal downward toward a low snapshot reference and would activate
only near a hold breach. This changes the controller objective and must be
discussed before implementation.
