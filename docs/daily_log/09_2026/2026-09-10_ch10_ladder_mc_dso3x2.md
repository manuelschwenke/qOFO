# 2026-09-10 - ch10 ladder Monte Carlo on the DSO_3 x2 network

## What was run

`ch_10_6_ladder_mc --runs 30 --jobs 10 --tag dso3x2` -> **39 accepted**
scenarios (overshoot from the in-flight wave), 70 % acceptance, **273 variant
runs**, `results/ch10_ladder_mc_dso3x2/`.  Event spread 1..5 per scenario;
30 of 39 carry two or more.

## What changed since the 2026-09-05 campaign

* **DSO_3 x2** on BOTH `dso_der_scale` and `dso_load_p_scale`, set in
  `make_thesis_config` (NOT in `make_config`, which stays the untouched
  authoritative factory).  Verified against the built net: DSO_3 DER
  700 -> 1400 MVA, load 261.8 -> 523.6 MW, DSO_1/2/4 unchanged.  Chapter 9's
  dead-band studies ADMIT only x2 runs, and the runner warns a scaled run is
  "NOT comparable with an unscaled run" -- the two chapters were describing
  different networks.
* **O3 no longer forces `coordination_mode="none"`.**  It inherits `sbx_h`
  like O2, so `O2->O3` isolates the DSO layer instead of adding it while
  simultaneously removing horizontal boundary exchange.
* **DS-side metrics added**: `ds_v_*` (terminal quality) and `ds_q_*`
  (diagnostic only).  `e_v` scores TS-zone voltage alone, and zone 1 carries
  no DSO, so the previous campaign could not see what the DSO rung buys.

## Result

O2->O3, paired over 39 scenarios: `e_v` -0.00099 (-12.9 %), **improving in
39/39**, closing 50 % of the O2->M gap; `e_v` spread 0.00286 -> 0.00239.
It is the ONLY step that improves in every scenario (O3->M: 95 %).
De-confounding raised it from -9.7 % / 95 % under the old setting.

DS voltage is a genuine trade: `ds_v_rms` +49.6 % (further from 1.03) while
band violations fall 65 % and the worst excursion is flat.  Which way that
reads depends on whether DS quality means "stay in band" or "hold 1.03".

## Findings to carry into the thesis

* **`L1->L2` is not a step**: -0.00013, improving in 38 %.  Third independent
  confirmation (deterministic run, 116 unscaled scenarios, 39 scaled).
* **DS voltage is NOT monotone down the ladder.**  S1 (0.01488, violation
  fraction 0.0215) and M (0.01662, 0.0077) are the WORST rungs on DS voltage.
  The centralised bound buys transmission performance by spending DS voltage
  hardest of all.
* `ds_q_*`, `e_p`, `tap_ops`, `avr_travel` are diagnostics, not objectives --
  now documented in `ds_metrics`' docstring so the distinction lives in code.

## Open

* The wired S1 log, the 90-min PI screen and both 360-min candidates were all
  produced UNSCALED; they no longer match the chapter-10 network.
* Extend to 100 with `--runs 100 --resume --tag dso3x2`.
* Worker count: 18 broke the pool in 4 min while a parallel session held ~19
  other Python processes (node-locked academic Gurobi licence the prime
  suspect); 10 ran a full day with zero restarts.
