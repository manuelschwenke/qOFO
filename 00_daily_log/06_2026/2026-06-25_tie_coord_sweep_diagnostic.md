# 2026-06-25 Tie-Coordination Parameter Sweep Diagnostic

Timestamp: 2026-06-25

Changed files:
- experiments/diag_tie_coord_sweep.py

Reason:
- Diagnose persistent L14 reactive tie-line exchange between zone 2 bus 9 and zone 1 bus 39 under the horizontal TSO-TSO voltage-corridor coordinator.
- Quantify the effect of feedforward smoothing beta (	ie_ff_smoothing), price anti-windup (	ie_lambda_max), and explicit tie-flow terms (g_z_q_tie, 	so_g_q_tie) without changing controller behaviour or default experiment logic.

Method / structure:
- Added a headless import-based diagnostic that loads experiments/000_M_TSO_M_DSO.py, modifies runtime config fields per case, suppresses per-run banners, and prints CSV metrics.
- Main reported quantities are L14 signed Q at the zone-1 endpoint, equivalent zone-2-to-zone-1 Q, endpoint voltages V39 and V9, lambda_e, zone-2 voltage limits/RMS error, and mean absolute tie-flow magnitude.

Preliminary result:
- Lower beta reduces L14 only marginally in the 8-minute smoke sweep; the dominant limitation appears to be the conflict between reducing bus-9 voltage and maintaining zone-2 voltage tracking at other buses.
- Strong explicit Q_tie penalties or soft-cap weights can reduce L14 substantially, but aggressive settings degrade zone-2 voltage feasibility and may cause non-convergent power flows.
