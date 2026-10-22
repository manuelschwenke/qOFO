# Autonomous local bank control, and confirmation on the regular path

Timestamp: 2026-10-21, 23:55 CEST.

Reason: the author asked for a three-way comparison — banks held, banks under
autonomous local voltage control, banks coordinated inside the MIQP — and, if
coordination loses, a fallback to the MIQP-only variant tuned to commit on
voltage deviations that are large *and* not recovered directly by the other
actuators. Both needed a piece that did not exist.

## Local autonomous control, as a change of driving signal

The integrating dispatcher already computes, per bank, its voltage sensitivity
column over its observed buses and the measured voltages there. The classic
substation rule is therefore a change of *driving signal*, not a new
controller:

    g_h = shunt_local_g · Σ_b D_db(V_b − V_ref) · h_col,b

with `D_db` the deadband operator. The device sign is carried by `h_col`, so a
capacitor releases and a reactor engages on high voltage without any extra
case analysis. The integrator's hysteresis, dwell, switching budget and voltage
guard are untouched, and the result is the accumulated-deviation switch of
CIGRE CSE 026 §2.6. Selected by `shunt_int_drive='local_voltage'`, which
excludes the interface and reserve terms from the drive and is rejected in
combination with `shunt_dispatch='hybrid'` or with STS support, since those also
replace the driving signal.

Added `shunt_local_g`, `shunt_local_v_ref_pu` (1.03, the zone setpoint) and
`shunt_local_v_deadband_pu` (0.01). Three study arms in
`experiments/shunt_hybrid_check.py`: `off` (banks installed, switching budget
zero), `local`, `coord`.

The gain was derived from a probe at unit gain on the selected parameter set:
over 168 bank-instants the drive has p50 = 3.5·10⁻⁷ and p90 = 2.1·10⁻⁶, so
readiness in three TS observations at the p90 deviation needs
`shunt_local_g ≈ 4.2·10⁵`. That is the value used in the comparison.

A caveat visible in those numbers: against a 1.03 reference with a 0.01 pu
deadband, the deviations that reach the bank's own buses barely exceed the
deadband, so the rule is operating close to its dead zone. A tighter deadband
or a per-bank reference would make it more active; 0.01 pu was kept because it
is the realistic substation value.

## Confirmation on the regular path

The regular path committed on the **first** instant its threshold was crossed,
which is why it fired at t = 20 s in the 20 October runs, eight minutes before
the disturbance. `shunt_hybrid_regular_confirmations` (default 1, the previous
behaviour) requires the MIQP to select the same bank and direction for that many
consecutive TS observations. A changed bank, a changed direction, an observation
gap, or an instant in which the MIQP holds every bank resets the count.

That last reset is the point: an instant in which the continuous actuators
resolve the deviation makes the MIQP stop asking for the bank, so the criterion
becomes *large enough to clear the movement weight* **and** *persistent enough
to outlast the continuous response* — which is the behaviour the author
specified for the fallback. When a proposal is unconfirmed the augmented problem
is re-solved with every bank held, so the ordinary actuators are dispatched
consistently rather than with the controls that had been optimised around a step
that is not being taken.

Verification: 174 tests pass with the one pre-existing skip. Five new cases
cover the default immediate commit, waiting for three confirmations while
dispatching the held solution, the reset when the deviation is recovered, and
rejection of the unsupported local-drive configurations.

## Outcome, 00:45 CEST

The three-way comparison and the campaign are in
`docs/tuning/2026-10-21_three_way_shunt_comparison.md`. In short:

- Coordination did **not** lose on the selected parameter set: against banks
  held it improves TS mean spatial voltage RMS by 1.9% and interface-Q RMSE by
  0.9%, costing 9% on the interface p95. The 54% interface degradation reported
  on 20–21 October was a property of the `comp_q_off` diagnostic set, whose
  subordinate interface objective is off and whose delegated request is frozen,
  and it does not transfer.
- Local autonomous control is the weakest arm on the interface (+13.8% RMSE,
  +35.8% p95) for the best STS spatial voltage RMS, which is the expected
  signature of an uncoordinated actuator.
- The campaign found the configuration the author specified:
  **`g_w_tso_shunt = 10` with two confirmations** — one switch per hour, on a
  sustained deviation after the disturbance, improving DER utilisation p95 by
  2.6%, TS spatial voltage RMS by 1.2% and STS by 4.5%, for 0.5% of interface-Q
  RMSE and 1.4% of its p95.
- At the derived threshold of 20, confirmation removes **every** commit: none of
  the three steps survives two consecutive instants, so by the author's own
  criterion all three were recovered by the other actuators anyway. The
  persistence test is what makes the lower threshold safe, so the two
  parameters are not independent knobs.

Open: the three-way comparison is running. If coordination loses, the fallback
is this path alone — supervisory weights zero — with `g_w_tso_shunt` from the
recorded gradient distribution (p99 = 33.8 on the selected set, with the usual
gap between p90 = 9.8 and p95 = 28.5) and the confirmation count as the second
knob.
