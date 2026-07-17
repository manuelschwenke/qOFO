# SBX-H voltage-tracking equity metric and 4x2 live plot

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** Manuel requested a quantitative measure of whether contractual
SBX-H support equalizes voltage-tracking accuracy across the three TSO areas,
and a fourth live-plot row beside cumulative bilateral settlement.

## Metric definition

For each area, all monitored TSO voltage errors over the latest
`k_sched` TSO samples are pooled. The per-area burden is

`E_z = 1000 * sqrt(mean((V_meas - V_ref)^2))` in mpu.

Every monitored bus/sample has equal weight inside its area; every area has
equal weight in the cross-area statistics. Tracking-burden inequality is the
finite-population-normalized Gini coefficient

`G_V = sum_z sum_j |E_z - E_j| / (2 * (N - 1) * sum_z E_z)`.

`G_V = 0` means equal area RMSE. The stored presentation-equivalent fairness
is `F_V = 1 - G_V`. If every area has zero tracking error, `G_V` is defined as
zero. Mean RMSE, worst RMSE, and the worst-area identifier are stored alongside
the Gini so perfect equality at a poor operating point cannot be mistaken for
good voltage tracking.

## Code changes

- `sbx_h/metrics.py`
  - reusable `VoltageTrackingEquity` result object;
  - validated per-area RMSE and normalized-Gini calculation.
- `sbx_h/adapter.py`
  - records an ex-post rolling error window of length `k_sched`;
  - uses all monitored TSO voltage outputs and their active references;
  - exposes `tracking_equity_history` for plots and later experiments;
  - does not feed the MIQP, settlement, escalation, or actuator dispatch.
- `visualisation/plot_sbx.py`
  - changes the three-corridor figure to a 4x2 GridSpec;
  - bottom-left panel shows each area's rolling RMSE and `G_V` on a bounded
    secondary axis;
  - the current mean, worst area/RMSE, and Gini are shown in the panel title;
  - bottom-right retains cumulative bilateral settlement.
- `tests/sbx_h/test_metrics.py`
  - equal, unequal, and all-zero burden cases.

## Verification

- Python compilation passed for the metric, adapter, and plot modules.
- `pytest -q tests/sbx_h`: **50 passed**.
- A 7-minute headless dynamic experiment completed 21 plant steps and saved
  the full 4x2 figure.
- Visual inspection of the full figure and bottom-row crop found no title,
  legend, center-label, or axis overlap.

## Assumptions and limits

- This is a TSO-layer tracking metric; underlying DSO/HV tracking should be
  evaluated separately rather than mixed into the same area burden.
- The live panel is a rolling contract-cycle diagnostic, not the final paired
  `none` versus `sbx_h` causal comparison. Thesis evaluation should reuse the
  metric over predefined identical disturbance windows in both arms.
- Equality alone is not sufficient. Mean RMSE, worst-area RMSE, voltage-bound
  violations, and OLTC operations remain necessary companion outcomes.
