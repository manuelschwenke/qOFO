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

(filled in after the run)
