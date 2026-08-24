# 2026-08-24 — Ch. 9.1 experiment A: worst settling over nodal voltages only

**Timestamp:** 2026-08-24 10:20 CEST

**What changed:** `experiments/ch_9_parameter_selection/ch_9_1_actuator_location_sweep.py`,
`tests/experiments/test_ch9_actuator_location_sweep.py`,
`docs/data/2026-08-24_ch9_actuator_location_voltage_only/` (new).

**Reason:** the reported worst settling time was reduced over every measured
output — 70 nodal voltages, 12 DSO interface-Q flows, 5 TSO boundary-Q flows —
and a Q flow won that reduction in **53 of 70 cases**, including the global
worst. Restrict the reduction to nodal voltages.

## Method

`WORST_OUTPUT_KINDS = ("voltage",)` now selects which `signal_metrics.csv` rows
the per-case reduction may pick; `reduce_worst(rows, kinds)` performs it and
raises on an empty scope rather than reporting 0 s for a case never judged.
Nothing about the *measurement* changed: every output kind is still recorded
with its own band, and the unrestricted reduction is retained per case as
`t_settle_all_s` / `worst_signal_all` / `censored_all`. `run_meta.json` records
both `worst_output_kinds` and `measured_output_kinds`, and `summary.md` states
the scope in the claim sentence and prints both global worsts.

## Result — no RMS was re-run

The case-level worst is a pure reduction over rows `signal_metrics.csv` already
held, so run `actuator_location_sweep_t0/20260822-014209` (70/70 complete,
2026-08-22) was re-aggregated from its own 6090 per-output rows. Originals are
untouched; the derived tables are in
`docs/data/2026-08-24_ch9_actuator_location_voltage_only/`.

| | all outputs | nodal voltages only |
|---|---|---|
| global worst case | `avr_p0p02pu_G10` | `avr_m0p02pu_G10` |
| global worst | 17.172 s at `Q_TSO_Z1_Z2_TN_line2` | **12.472 s** at `V_TN_bus1` |
| censored cases | 0 / 70 | 0 / 70 |

Settling falls in 50 of 70 cases (it can only fall). The binding actuator class
is unchanged — AVR `v_ref` — with the TSO gen-transformer OLTC next at 11.812 s
(was 15.262 s). Per-class deltas range from 0.000 s (`dso_der_q` +1, a sub-band
case) to 7.900 s (`avr_vref` +1).

## Effect on the admissible rates

The worst RMS settling the DSO layer must clear drops 17.172 s → 12.472 s, i.e.
**−27 %**. Both figures sit below the configured `dso_period_s = dt_s = 20 s`,
so **the selected rates do not change**: 20 s already cleared 17.172 s. What
changes is the margin, 1.16× → 1.60×. The TSO period (180 s) was never near
binding on this quantity.

## Assumptions

- Bands are unchanged: 1 mpu on voltage, 1 Mvar on Q, 5 s confirmation window,
  final-window mean as the reference value.
- Settling is measured from the dispatch instant and includes the OLTC's 5 s
  command-to-contact delay.
- Snapshot `full_t0_20160105-0800.json`; disturbances and currents excluded.

## Risks / unresolved

1. **The narrowed claim no longer covers interface Q.** Q at the EHV–HV
   interface and at the zone boundaries is a *controlled output* of Layer 2 and
   Layer 1 respectively (it is the tracked quantity, not a by-product), so
   timescale separation argued on voltages alone does not establish that the
   inner loop's Q response has settled before the outer loop acts. The 17.172 s
   figure is retained per case precisely so this can be argued explicitly rather
   than by omission.
2. **The stated motivation for narrowing is band comparability, and it is an
   argument, not a measurement.** The 1 Mvar band is absolute while the Q steps
   differ by more than an order of magnitude across the 12 DSO interfaces and
   5 TSO corridors, so a Q row can win on band tightness rather than corridor
   speed. A per-corridor relative band (e.g. a fraction of that corridor's own
   step) would test this directly and is not implemented.
3. The reduction scope is a module constant, not a CLI flag, deliberately: a
   flag would let an archived run silently differ from the documented claim.
   Changing it requires editing the script, which the per-run `script_snapshot.py`
   then pins.
