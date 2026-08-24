# Actuator-only settling-time location sweep

Status: **complete** (70/70 cases complete).

## Design

- 46 feasible discrete steps are exhaustive in RMS.
- 24 continuous cases are RMS-verified after deterministic QSS screening.
- Controlled outputs: 70 nodal voltages, 12 DSO interface-Q flows, and 5 TSO boundary-Q flows.
- Currents and disturbances are excluded.
- Fixed bands: 0.001 pu and 1 Mvar; confirmation window 5 s.
- OLTC: 5 s pure delay followed by Tmech=0.05 s; settling starts at dispatch.
- Each case pre-settles for 300 s and starts at 30 s, extending to 120 s if censored.

The claim supported by this run is the worst RMS settling time among all feasible discrete locations and the deterministically screened continuous locations. It is not a mathematical proof over untested continuous locations.

## Current worst by actuator class and direction

| Class | Direction | Case | Settling [s] | Output | Flag |
|---|---:|---|---:|---|---|
| avr_vref | -1 | `avr_m0p02pu_G10` | 15.722 | Q_TSO_Z2_Z1_TN_line14 | settled |
| avr_vref | 1 | `avr_p0p02pu_G10` | 17.172 | Q_TSO_Z1_Z2_TN_line2 | settled |
| coupler_oltc | -1 | `tap_m1_NC3W_DSO_3_t7` | 7.412 | Q_TSO_Z1_Z2_TN_line2 | settled |
| coupler_oltc | 1 | `tap_p1_NC3W_DSO_1_t0` | 5.112 | Q_DSO_NC3W_DSO_1_t0 | settled |
| dso_der_q | -1 | `q_dso_der_q_m33Mvar_DER_DSO_3_s25_b88` | 3.472 | Q_TSO_Z1_Z2_TN_line2 | settled |
| dso_der_q | 1 | `q_dso_der_q_p41Mvar_DER_DSO_4_s35_b108` | 0.072 | V_DSO_4_bus108 | settled |
| shunt | 1 | `shunt_p1_SH_MSR_DSO_3_s5` | 3.372 | Q_TSO_Z1_Z2_TN_line2 | settled |
| tso_der_q | -1 | `q_tso_der_q_m60Mvar_WP_TSO_s2_b24` | 9.372 | Q_TSO_Z2_Z1_TN_line14 | settled |
| tso_der_q | 1 | `q_tso_der_q_p60Mvar_WP_TSO_s2_b24` | 8.322 | Q_TSO_Z1_Z2_TN_line2 | settled |
| tso_oltc | -1 | `tap_m1_MT_g0_t0` | 15.262 | Q_TSO_Z1_Z2_TN_line2 | settled |
| tso_oltc | 1 | `tap_p1_MT_g0_t0` | 14.512 | Q_TSO_Z1_Z2_TN_line2 | settled |

Current global worst: `avr_p0p02pu_G10`, 17.172 s at `Q_TSO_Z1_Z2_TN_line2`.
