# DSO_2 voltage margin: quick test of three remedies (W1)

Timestamp: 2026-10-25, ~18:30 Europe/Berlin.

Reason: with the selected weights (c08b1d3d33ab) DSO_2 reaches 1.096 pu in W1
(`2026-10-25c_...`). Author asked to test an earlier DSO_2 tap engagement, and
proposed a DSO operating band with a strategic margin of 0.02 pu
(0.92–1.08 pu) enforced through the voltage slack.

## Method

Diagnostic only, no repository change:
`results/THESIS_ch8_weight_selection_20261024/variant_unit.py`, W1, 12 h,
same construction as the Stage-II runs, records kept (`dso2_quicktest/`).
`tap`: DSO_2 coupler tap price 4.627 → 3.639 (engagement at the designed
0.015 pu instead of ~0.019). `relief`: DSO_2 `dso_g_v` and tap price ×4.574
(voltage priority back to the Stage-I level, tap engagement unchanged).
`margin`: every DSO controller's band 0.92–1.08 pu (class default 0.90–1.10),
injected by a `DSOControllerConfig` subclass in the test process; slack weight
`g_z_voltage` 1e9 unchanged. λ_max(B) for STS_DSO_2: 1.249 (selected), 1.25
(tap), 1.289 (relief).

## Result (W1)

| variant | e_v(TS) | DSO_2 V_max / margin / time > 1.08 | DSO_4 V_max / margin | DSO_2 tracking RMS [Mvar] |
|---|---|---|---|---|
| selected | 0.00443 | 1.096 / 0.004 / 11.6 % | 1.089 / 0.011 | 2.25 |
| tap | 0.00444 | 1.089 / 0.011 / 15.8 % | 1.089 / 0.011 | 3.37 |
| relief | 0.00443 | 1.063 / 0.037 / 0 % | 1.089 / 0.011 | 3.23 |
| margin | 0.00443 | 1.078 / 0.022 / 0 % | 1.082 / 0.018 | 2.60 |
| Stage-I point | 0.00527 | 1.046 / 0.054 / 0 % | 1.069 / 0.031 | 12.85 |

- `tap` barely helps (the problem is a spread inside DSO_2, not an offset the
  tap engagement reads) and lengthens the time above 1.08 pu.
- `relief` fixes DSO_2 (e_v(DS) of DSO_2 0.0119 → 0.0065) but not DSO_4, and is
  a per-area weight change after the selection.
- `margin` keeps DSO_2 at 1.078 and DSO_4 at 1.082 (0.8 % of the time above
  1.08: the band acts on the linear prediction), with e_v(TS) unchanged and a
  small tracking cost. It is an operating rule applied to all DSOs alike.

## Open, for the author

- The band needs a real configuration field (DSO voltage band passed to
  `DSOControllerConfig`; today the runner never sets it). Then either verify
  the selected set with it (W1–W6 + settling, ~2–3 h, stated as introduced
  after the selection) or re-apply the procedure with it (~40 h).
