# Amendment 1 to the weight-selection procedure: STS coupler engagement inside the operating band

**Status:** declared 2026-10-28, before execution. Author decision 2026-10-28
for option (c1) of `2026-10-28_tap_engagement_at_selected_weights.md`.

**Amends:** `2026-10-25_weight_selection_procedure.md` (frozen 2026-10-25,
applied once in run 0001; result `ch9_selected_band_2026_10` =
fde2e81af6b1). The as-frozen result is kept and reported. This amendment
was made **after** that result was known, and the thesis must say so.

**Producer:** `experiments/ch_09_parameter_selection/ch_9_3_amendment1.py`.

## Rule

The filter of the selection (step 1 of the procedure) gets one more
condition:

$$
\theta_{\mathrm{STS,min}}
= \min_{c \in \{W1..W6\}} \min_{i \in \mathrm{couplers}}
\frac{g_{w,\mathrm{dso\_oltc}}}{2\, g_{V,\mathrm{DS}}\, \big|\sum_j H_V^{(c)}[j,i]\big|}
\;\le\; V_{\max,\mathrm{band}} - V_\mathrm{ref} = 1.08 - 1.03 = 0.05\ \mathrm{pu}.
$$

$H_V^{(c)}$ comes from the operating points the campaign loaded
(`results/stage1/operating_points/op_rural_700_net82afed5b_W*.npz`), and θ is
evaluated with `tuning.stage1_screen.engage_deviation`, the Stage-I
machinery.

**Rationale.** The quantity θ_STS,min is the uniform deviation from the
reference at which the strongest STS coupler commits one step on the voltage
gradient. If it exceeds the headroom of the operating band, no coupler can
act on the voltage error while the STS is inside the band; it moves only
when the soft band constraint becomes active. The bound uses only the band
(set 2026-10-25) and the reference (1.03 pu), both fixed before the campaign.
It does not refer to any result of the campaign.

Everything else is unchanged:
- the remaining filter limits;
- the ranking (e_v(TS), then e_v(DS) within 2 %);
- the settling requirement (p95 ≤ 27, median residual ≤ 1 Mvar), in rank
  order.

## Application

The rule is applied to the 242 candidates evaluated in run 0001, with no new
Stage-II simulation: the Stage-II-b path stays the one the as-frozen ranking
produced.
1. Admitted candidates are passed to the unchanged `ch_9_3_select.py` with
   the campaign's arguments.
2. Experiment C runs in the new rank order. Measured settling results of run
   0001 are reused, since they come from the same campaign and closed loop.
   The aborted run of 301c285ee52b is not a result and is repeated.
3. The result is written as `configs/paramsets/ch9_selected_band_2026_10_a1.json`.
   `SELECTED_PARAMSET` is switched only by a separate author decision.

**Check of intent (added by the author, 2026-10-28).** The amendment is
meant to make the tap changers engage when many STS voltages (the
area-mean/median, as in the live plot) are far from 1.03 pu. The producer
therefore re-simulates W1–W4, W6 for the amended rank 1 and 2, the as-frozen
result and the Stage-I point, with a read-only probe of every tap's commit
ratio. If the amended result does not show this behaviour, the amendment
fails its purpose. The remedy is then a re-application with a different
strategy (option (b) of the decision note), not a further amendment.

## Execution record

Run `results/ch_09_parameter_selection/0008_2026-10-28_103321_ch_9_3_amendment1/`,
started 2026-10-28 10:33 on commit dddfa45.

**Selection** (10:34). Of the 242 candidates of run 0001:
- 140 satisfy the bound; 51 of these are feasible;
- unchanged ranking: 1. **301c285ee52b**, 2. c08b1d3d33ab,
  3. 67ad1cf843ac, 4. 9e48320e8a01.

**Replay of the Stage-II path** (`analysis/tap_engagement/replay_path.py`,
`replay_path.log`). The amended rule picks the same incumbent as the frozen
rule after II-a (faf088d2570e) and after rounds 1–3 (e552b328fd76,
ff3f747d3f96, c08b1d3d33ab). Every round finds a new best, so the span factor
stays at 0.5 and the stage ends at the 4-round cap. An application of the
amended procedure from scratch would therefore:
- draw the same seeded points around the same centres;
- evaluate the same 242 candidates;
- differ from run 0001 only in the final selection, given bit-exact
  simulations.

**Engagement probe** (W1–W4, W6, 12 h; `probe/`, `engagement_eval.log`,
`analysis/tap_engagement/engagement_eval.py`). All 20 units reproduce the
stored e_v(TS) exactly (difference 0), so the probe does not disturb the
closed loop.

The deviation is that of the median of the measured STS voltages from
1.03 pu. It is summed over 4 areas × 5 windows (240 area-hours). A far
episode is |median dev| ≥ 0.02 pu for ≥ 10 min.

| | STS commits | min ≥ 0.010 | ≥ 0.015 | ≥ 0.020 | ≥ 0.025 | far episodes (served) |
|---|---|---|---|---|---|---|
| **301c285ee52b** (amended #1) | 22 | 949 | 25 | 1 | 0 | 0 |
| c08b1d3d33ab (amended #2) | 8 | 3 328 | 292 | 16 | 0 | 1 (0) |
| fde2e81af6b1 (as frozen) | 1 | 6 792 | 3 327 | 782 | 78 | 16 (0), 723 min |
| 3cb66225e1a7 (Stage I) | 19 | 2 170 | 50 | 1 | 0 | 0 |

**Commits of 301c285ee52b.** Of the 22:
- 9 follow a disturbance and 8 fall in the 30-min start-up;
- all are +1, i.e. voltage-lowering; there is no reversal and no hunting;
- at most 2 per coupler per 12 h.

The 5 other commits are all gradient-driven (commit ratio 1.001–1.111). They
occur at a median deviation of 0.009–0.015 pu (mean 0.011–0.016 pu).

The one commit at median < 0.010 pu (W1, DSO_4|trafo_10, 11.02 h) still has
a mean deviation of 0.011 pu and v_max 1.076 pu. The commit condition weighs
the buses by sensitivity, not by the median.

The single commit of the as-frozen set is constraint-driven (ratio 0.27,
v_max 1.0795 pu).

**Settling** of 301c285ee52b (Experiment C, 8 workers): running, 24/120
steps at 11:45.
