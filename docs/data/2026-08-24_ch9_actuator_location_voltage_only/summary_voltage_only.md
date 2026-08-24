# Re-aggregation: worst settling over nodal voltages only

Derived from `signal_metrics.csv` of `20260822-014209` on 2026-08-24. No RMS was
re-run: the per-output rows were already recorded, so restricting the
reduction to `output_kind == "voltage"` is a re-aggregation of the same
measurements. `results.csv` and `summary.md` are the unmodified originals.

## Effect

- Cases: 70. The all-output worst was an interface-Q flow in **53 of 70**.
- Settling changes in 50 cases; it can only fall.
- Censored cases: 0 (all outputs) -> 0 (voltages only).
- **Global worst: `avr_m0p02pu_G10`, 12.472 s at `V_TN_bus1`** (all outputs: `avr_p0p02pu_G10`, 17.172 s at `Q_TSO_Z1_Z2_TN_line2`).

## Worst by actuator class and direction

| Class | Dir | Case (voltages) | Settling [s] | Output | All-output [s] | Delta [s] |
|---|---:|---|---:|---|---:|---:|
| avr_vref | -1 | `avr_m0p02pu_G10` | 12.472 | V_TN_bus1 | 15.722 | 3.250 |
| avr_vref | 1 | `avr_p0p02pu_G10` | 9.272 | V_TN_bus1 | 17.172 | 7.900 |
| coupler_oltc | -1 | `tap_m1_NC3W_DSO_1_t2` | 6.112 | V_DSO_3_bus92 | 7.412 | 1.300 |
| coupler_oltc | 1 | `tap_p1_NC3W_DSO_1_t0` | 5.062 | V_DSO_1_bus43 | 5.112 | 0.050 |
| dso_der_q | -1 | `q_dso_der_q_m23p1Mvar_WPC_DSO_2_s21_b63` | 1.922 | V_TN_bus28 | 3.472 | 1.550 |
| dso_der_q | 1 | `q_dso_der_q_p41Mvar_DER_DSO_4_s35_b108` | 0.072 | V_DSO_4_bus108 | 0.072 | 0.000 |
| shunt | 1 | `shunt_p1_SH_MSR_DSO_4_s7` | 2.972 | V_TN_bus1 | 3.372 | 0.400 |
| tso_der_q | -1 | `q_tso_der_q_m60Mvar_WP_TSO_s3_b5` | 4.922 | V_TN_bus28 | 9.372 | 4.450 |
| tso_der_q | 1 | `q_tso_der_q_p60Mvar_WP_TSO_s2_b24` | 2.922 | V_TN_bus27 | 8.322 | 5.400 |
| tso_oltc | -1 | `tap_m1_MT_g1_t2` | 11.812 | V_DSO_3_bus87 | 15.262 | 3.450 |
| tso_oltc | 1 | `tap_p1_MT_g0_t0` | 9.862 | V_TN_bus1 | 14.512 | 4.650 |

Interface Q is a controlled output of both layers, so this bound does
NOT cover it; `t_settle_all_s` is the column that still does.
