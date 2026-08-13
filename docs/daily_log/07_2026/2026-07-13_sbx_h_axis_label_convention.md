# SBX-H axis-unit convention and equity right-axis spacing

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** Manuel requested the thesis-style `quantity / unit` convention
for every SBX-H axis label and an explicitly right-positioned Gini-axis label
without overlap with cumulative bilateral settlement.

## Changes

- Replaced bracketed units with slash-form labels:
  - `Time / min`;
  - `dQ, Q_sup / Mvar`;
  - `Cycle RMSE / mpu`;
  - `Net payment / EUR`.
- Retained the already conforming `V / pu` label.
- Expressed the dimensionless normalized Gini as `G_V / 1`.
- Explicitly pinned the Gini ticks, spine, and label to the right side of the
  voltage-tracking-equity panel.
- Increased horizontal subplot spacing from `0.18` to `0.24` so the equity
  right-axis label and settlement left-axis label have separate center space.

## Scope

This is a presentation-only change. Metrics, controller objectives,
constraints, actuators, SBX-H settlement, and recorded outputs are unchanged.
