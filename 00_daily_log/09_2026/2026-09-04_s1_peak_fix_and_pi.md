# S1 pilot-node controller: voltage-priority fix and optional PI paths

Timestamp: 2026-09-04 07:38 CEST (Europe/Berlin).

Reason: remove the post-contingency voltage-error growth diagnosed in
`2026-09-03_s1_pilot_peak_diagnosis.md`, correct generator-outage handling, and
investigate whether proportional action should be added to the regional voltage
regulator (RVR) and plant reactive-power regulator (RPR).

## Scope and decision

The controlled outputs remain the three S1 pilot-node voltages and the
machine/TS-DER reactive-power sharing targets. The actuators remain generator AVR
references and direct TS-DER Q commands. Machine-transformer taps remain held,
switched shunts remain disabled, and the DSO coupler taps remain under their local
relays. The RVR and RPR integral grids remain 180 s and 20 s, respectively.

S1 now enables a voltage-priority condition for generator RPR moves. Optional
parallel PI paths were added to both RVR and RPR, but their gains remain zero in
S1 and in the global defaults. The 42-minute screening runs do not justify a
nonzero production gain: an inner P term amplified the target disturbance, and
the outer P result depended materially on its sampling grid. A separate tuned,
full-horizon PI study is therefore required before changing the S1 gains.

## Code changes

- `controller/svr_controller.py`
  - Generator dispatchability is now the intersection of the static controller
    mask and `net.gen.in_service`. An unavailable generator contributes neither
    measured Q nor reactive headroom, its AVR reference is frozen, and its RPR
    state is reinitialised bumplessly when it returns.
  - `rpr_voltage_priority` blocks a candidate machine RPR move when the product
    of cached `dV_pilot/dQ`, cached `dQ/dV_set`, and the proposed AVR-reference
    increment predicts motion opposite to the measured pilot-voltage error. A
    blocked move also rolls back the hidden RPR integrator state.
  - RVR and RPR now support nonnegative `k_p_rvr` and `k_p_rpr`. Their integral
    states are separate from the algebraic P outputs. Conditional integration
    prevents windup at the `q_level` and AVR-reference limits. With both gains
    zero, the controller reproduces the staged I-only trajectory exactly.
  - The outer P output is held on the existing 180 s RVR grid by default. The
    explicit `rvr_p_every_step` option evaluates only that algebraic path on the
    20 s call grid while retaining the 180 s integral update.
- `configs/config.py`: adds the two P gains, fast-P switch, and voltage-priority
  switch with zero/false defaults.
- `experiments/runners/multi_tso_dso.py` and
  `experiments/helpers/records.py`: wire the options and store Q references,
  availability, block flags, saturation, and both P contributions.
- `experiments/ch_10_case_study/ch_10_1_variant_ladder.py`: enables
  `svr_rpr_voltage_priority=True` for S1; both P gains remain at zero and the fast
  P switch remains false.
- `tests/test_svr_controller.py`: adds focused outage, restoration, guard, PI,
  anti-windup, sampling, and bumpless-transfer regressions.

## Implemented PI equations

For area `a`, at an RVR measurement instant,

\[
x_a^+ = x_a + \frac{\Delta t_{\mathrm{RVR}}}
 {T_{\mathrm{RVR}}\lvert s_{p,a}\rvert}
 (v_{p,a}^{\mathrm{ref}}-v_{p,a}),\qquad
q_a = \operatorname{sat}_{[-1,1]}\!\left(
x_a + \frac{K_{p,\mathrm{RVR}}}{s_{p,a}}
(v_{p,a}^{\mathrm{ref}}-v_{p,a})\right).
\]

For an available synchronous machine `i`, at an RPR instant,

\[
y_i^+ = y_i + \frac{\Delta t_{\mathrm{RPR}}}{T_{\mathrm{RPR}}}
\frac{Q_i^{\mathrm{ref}}-Q_i}{\partial Q_i/\partial v_i^{\mathrm{set}}},
\qquad
v_i^{\mathrm{set}} = \operatorname{sat}\!\left(
y_i + K_{p,\mathrm{RPR}}
\frac{Q_i^{\mathrm{ref}}-Q_i}{\partial Q_i/\partial v_i^{\mathrm{set}}}
\right).
\]

Only the states `x_a` and `y_i` receive sample-time factors. The signed cached
local slope is retained in the RPR normalisation. In the present benchmark all
active `dQ/dV_set` slopes are positive.

## Validation

The original controller was first reproduced from the same archived S1 case.
Across the 48-minute replay, its maximum differences from the archived run were
`2.64e-13 pu` in the plotted RMS metric and `1.54e-12 pu` in any bus voltage.
The zero-gain PI staging trajectory was then compared sample-for-sample against
the staged I-only controller, including outage and restoration, with exact
equality. The final deployable unit suite reports 21 passed tests.

The selected voltage-priority/outage revision was simulated over the full
360-minute, 1080-record S1 scenario:

| Metric | Original S1 | Revised S1 | Change |
|---|---:|---:|---:|
| time RMS voltage error | 0.0123425 pu | 0.0119198 pu | -3.42% |
| generator-trip peak | 0.0314245 pu | 0.0153071 pu | -51.29% |
| growth after first trip sample | 0.0149378 pu | 0.0008790 pu | -94.12% |
| line-trip peak | 0.0273831 pu | 0.0166952 pu | -39.03% |
| minimum TSO voltage | 0.946584 pu | 0.947961 pu | +0.001377 pu |
| maximum TSO voltage | 1.059528 pu | 1.055570 pu | -0.003957 pu |

The tripped generator's AVR reference is held at 1.038920 pu throughout its
30--180 minute outage; the original implementation drove the offline reference
to its 1.05 pu ceiling. Across the full revised run, 42.16% of available
machine-update opportunities were blocked. This is an intentional Q-tracking
trade-off: pilot-voltage support takes priority while the regional level is held.

PI screening used the same plant and the first 42 minutes of the same scenario:

| Controller change | time RMS / pu | generator-trip peak / pu | Relative to original peak |
|---|---:|---:|---:|
| original I/I | 0.0097900 | 0.0314245 | -- |
| voltage priority, Kp=0 | 0.0075265 | 0.0153071 | -51.29% |
| outer P=0.05, held 180 s | 0.0093187 | 0.0277460 | -11.71% |
| outer P=0.20, held 180 s | 0.0106306 | 0.0217736 | -30.71% |
| outer P=0.20, evaluated 20 s | 0.0083289 | 0.0222102 | -29.32% |
| inner P=0.20 | 0.0098989 | 0.0323391 | +2.91% |
| voltage priority + outer P=0.20 | 0.0072412 | 0.0137888 | -56.12% |
| voltage priority + both P=0.20 | 0.0071866 | 0.0136374 | -56.60% |

The held outer P=0.20 case lowers the trip peak but creates larger oscillatory
peaks during initial handover and raises the 42-minute RMS by 8.59%. Evaluating
the P path on the 20 s grid avoids most of that penalty, but changes the
supervisory cadence. The inner P term accelerates the harmful Q-restoration
mechanism when voltage priority is absent. The small additional benefit of P on
top of the guard has only been tested over 42 minutes.

## Literature interpretation

Corsi's published Italian RVR description and the RTE account of French SVR both
support PI action in practical pilot-node schemes. The RTE paper also documents
the same nonminimum-phase contingency response found here: plant reactive-power
regulators can initially drive voltage in the wrong direction while restoring Q.
The SEST 2024 paper describes a parallel, rather than cascaded, SVR arrangement;
its equation does not establish the sensitivity-normalised cascaded equations
used here. The cited PSCC 2022 paper concerns local distribution-network DER PI
control and supports only generic discrete PI/anti-windup practice, not these
RVR/RPR formulas.

- https://doi.org/10.1109/PowerTech55446.2023.10202952
- https://studylib.net/doc/18816652/the-secondary-voltage-regulation-in-italy
- https://doi.org/10.1109/SEST61601.2024.10694346
- https://pscc-central.epfl.ch/repo/papers/2022/20300.pdf

## Remaining work and thesis consistency

`fig:case3:vtrack` and `tab:case3:ladder` still contain the archived classical
S1 run. They were deliberately not overwritten in this code change: enabling the
guard changes S1 from the unmodified classical reference to a documented
voltage-priority extension. Before regenerating the thesis artefacts, revise the
S1 definition and the Chapter 8 I-only claims, then rerun the entire ladder so
the S1--O1 comparison has one consistent provenance.

Suggested Obsidian update: add a note titled **S1 voltage-priority extension and
PI sampling screen**, link it to the classical SVR reference note and the
2026-09-03 peak diagnosis, and track full-horizon gain tuning as a separate
experiment rather than adopting `Kp=0.2` from the short screening run.
