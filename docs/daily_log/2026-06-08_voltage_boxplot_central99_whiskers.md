# 2026-06-08 — Voltage box plots: central-99.8% whiskers + per-variant V band table

## What changed
`experiments/006_CIGRE_MONTECARLO.py` — the two **voltage-spread** box plots now draw
whiskers at the **central-99.8% interval (0.1th / 99.9th percentile)** instead of the
matplotlib default Tukey 1.5·IQR fences.

- `plot_voltage_zone_boxplots` (Fig_mc_voltage_zone_box): added `whis=(0.1, 99.9)` to
  the `ax.boxplot(...)` call.
- `plot_paper_combined_box` panel 2 / `ax2` (Fig_mc_paper_combined): added
  `whis=(0.1, 99.9)` to the `ax2.boxplot(...)` call.

(History: first set to `(0.5, 99.5)` = central-99%, then revised to `(0.1, 99.9)` =
central-99.8% at the user's request.)

Panels 1 & 3 of the combined figure (RMS ΔV TS, gen Q-utilisation) were **left on the
default Tukey whiskers** — they are per-run *scalar* metrics (n=100), where the
extreme percentiles collapse onto the data extremes and the IQR-based outlier view is
the correct convention.

`showfliers=False` is unchanged, so nothing is drawn beyond the new whisker caps.

## Why
Whisker caps are now a fixed, reportable quantile of the pooled per-zone EHV
voltage envelope, so the figure whiskers line up 1:1 with the per-variant voltage
band reported in the paper table. User chose the central-99.8% band.

## Reproduce / regenerate
Figure-only change — rebuild with `python experiments/006_CIGRE_MONTECARLO.py --replot`
(no re-simulation needed). The pooled distribution per (variant, zone) is the
concatenation of the per-step `Vz__{V}__{zone}__{min,mean,max}` arrays from
`results/006_cigre_mc/timeseries/run_*.npz` (unchanged).

## Side artifacts (diagnostic, not part of the pipeline)
- `experiments/diag_extract_vminmax.py` — standalone extractor: per-(variant, zone)
  abs min/max, Tukey whiskers, q25/median/q75, and p0.1/p0.5/p2.5/p97.5/p99.5/p99.9;
  plus a per-variant worst/best-of-zones central-99.8% summary.
- Outputs: `results/006_cigre_mc/voltage_min_max_per_variant.csv` (per zone) and
  `voltage_central998_per_variant.csv` (per-variant worst/best central-99.8%).
  (`voltage_central99_per_variant.csv` from the earlier 99% pass may also still exist.)

## Per-variant central-99.8% band (worst/best across zones, p.u.)
| V | low p0.1 (zone) | high p99.9 (zone) | abs min | abs max |
|---|---|---|---|---|
| V1 | 0.9016 (z3) | 1.0793 (z1) | 0.7947 | 1.0819 |
| V2 | 0.9018 (z3) | 1.0793 (z1) | 0.8273 | 1.0819 |
| V3 | 0.9091 (z2) | 1.1049 (z3) | 0.8303 | 1.1216 |
| V4 | 0.9532 (z3) | 1.0858 (z3) | 0.9347 | 1.1064 |
| V5 | 0.9551 (z3) | 1.0759 (z3) | 0.9407 | 1.0887 |
