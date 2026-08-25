# Worst settling by output family: nodal voltage and interface Q

Derived from `signal_metrics.csv` of `20260822-014209` on 2026-08-25. No RMS was
re-run: the per-output rows were already recorded, so reducing them per
family re-reads the same measurements. `results.csv` and `summary.md` are
the unmodified originals.

Voltage and interface Q are reduced **separately, against their own bands**
(0.001 pu and 1 Mvar). They are not
pooled: the two are not commensurable, and a single max hides which family
bound the result.

## Global worst

| Family | Case | Settling [s] | Output |
|---|---|---:|---|
| nodal voltage | `avr_m0p02pu_G10` | 12.472 | V_TN_bus1 |
| interface Q (both layers) | `avr_p0p02pu_G10` | 17.172 | Q_TSO_Z1_Z2_TN_line2 |
| - DSO interface Q | `tap_p1_MT_g1_t2` | 8.412 | Q_DSO_NC3W_DSO_3_t7 |
| - TSO boundary Q | `avr_p0p02pu_G10` | 17.172 | Q_TSO_Z1_Z2_TN_line2 |

Q settles later than V in **50 of 70** cases.
Censored: 0 (V), 0 (Q), out of 70.

## Worst by actuator class and direction

The two halves of a row are reduced independently, so they may be
different cases. `(c)` = censored, `(sb)` = never left its band.

| Class | Dir | Case (worst V) | V [s] | V output | Case (worst Q) | Q [s] | Q output |
|---|---:|---|---:|---|---|---:|---|
| avr_vref | -1 | `avr_m0p02pu_G10` | 12.472 | V_TN_bus1 | `avr_m0p02pu_G10` | 15.722 | Q_TSO_Z2_Z1_TN_line14 |
| avr_vref | 1 | `avr_p0p02pu_G10` | 9.272 | V_TN_bus1 | `avr_p0p02pu_G10` | 17.172 | Q_TSO_Z1_Z2_TN_line2 |
| coupler_oltc | -1 | `tap_m1_NC3W_DSO_1_t2` | 6.112 | V_DSO_3_bus92 | `tap_m1_NC3W_DSO_3_t7` | 7.412 | Q_TSO_Z1_Z2_TN_line2 |
| coupler_oltc | 1 | `tap_p1_NC3W_DSO_1_t0` | 5.062 | V_DSO_1_bus43 | `tap_p1_NC3W_DSO_1_t0` | 5.112 | Q_DSO_NC3W_DSO_1_t0 |
| dso_der_q | -1 | `q_dso_der_q_m23p1Mvar_WPC_DSO_2_s21_b63` | 1.922 | V_TN_bus28 | `q_dso_der_q_m33Mvar_DER_DSO_3_s25_b88` | 3.472 | Q_TSO_Z1_Z2_TN_line2 |
| dso_der_q | 1 | `q_dso_der_q_p41Mvar_DER_DSO_4_s35_b108` | 0.072 | V_DSO_4_bus108 | `q_dso_der_q_p41Mvar_DER_DSO_3_s25_b88` | 0.072 | Q_DSO_NC3W_DSO_3_t8 |
| shunt | 1 | `shunt_p1_SH_MSR_DSO_4_s7` | 2.972 | V_TN_bus1 | `shunt_p1_SH_MSR_DSO_3_s5` | 3.372 | Q_TSO_Z1_Z2_TN_line2 |
| tso_der_q | -1 | `q_tso_der_q_m60Mvar_WP_TSO_s3_b5` | 4.922 | V_TN_bus28 | `q_tso_der_q_m60Mvar_WP_TSO_s2_b24` | 9.372 | Q_TSO_Z2_Z1_TN_line14 |
| tso_der_q | 1 | `q_tso_der_q_p60Mvar_WP_TSO_s2_b24` | 2.922 | V_TN_bus27 | `q_tso_der_q_p60Mvar_WP_TSO_s2_b24` | 8.322 | Q_TSO_Z1_Z2_TN_line2 |
| tso_oltc | -1 | `tap_m1_MT_g1_t2` | 11.812 | V_DSO_3_bus87 | `tap_m1_MT_g0_t0` | 15.262 | Q_TSO_Z1_Z2_TN_line2 |
| tso_oltc | 1 | `tap_p1_MT_g0_t0` | 9.862 | V_TN_bus1 | `tap_p1_MT_g0_t0` | 14.512 | Q_TSO_Z1_Z2_TN_line2 |

## Reading it

The voltage column and the Q column answer different questions and both
must be quoted. The Q band is a fixed absolute 1 Mvar against flows whose
own step sizes differ by more than an order of magnitude across the 12 DSO
interfaces and 5 TSO corridors, so a Q worst can be set by a corridor with
a small step rather than by a slow one; the voltage band is a network
quantity with the same meaning at every bus. Neither observation makes the
other family droppable: interface Q is the tracked output of both layers.
