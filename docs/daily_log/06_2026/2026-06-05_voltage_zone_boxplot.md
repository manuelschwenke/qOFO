# 2026-06-05 — Per-zone TS voltage-range boxplot (006)

**Timestamp:** 2026-06-05 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
User request: a boxplot of TS bus voltages [p.u.] with 15 boxes = 5 variant
groups (V1–V5) x 3 boxes (one per TS zone), to compare how tightly each control
variant holds voltages per zone across the Monte-Carlo ensemble.

## What was changed (`experiments/006_CIGRE_MONTECARLO.py`)
- **`extract_timeseries`** now also returns `zv`: per TS zone, the per-step
  voltage envelope `{'min','mean','max': array[n_steps]}` read from the record
  fields `zone_v_min/mean/max` (the voltage-observed EHV buses per zone).
- **`_write_ts_npz`** gained an optional `zv_by_variant` arg and persists it as
  npz keys `Vz__{variant}__{zone}__{min|mean|max}`.
- **`run_one_scenario`** wires `zv` through (`zv_by_variant`).
- **New `plot_voltage_zone_boxplots(runs)`** (placed before
  `make_aggregate_figures`, added to its render loop): grouped boxplot, 5 variant
  groups x 3 zone boxes. Each box pools the per-zone min/mean/max envelope over
  all steps and all runs, so whiskers show the voltage range; dashed line at
  `V_SET=1.03`; zone legend. Writes `Fig_mc_voltage_zone_box.pdf/.png` to
  `OUT_ROOT` and mirrors the PDF to `PAPER_FIG_DIR`. Skips gracefully (with a
  re-run hint) if the npz predate the `Vz__` keys.

Note: added as a dedicated function, NOT inside `plot_table3_boxplots` — the
latter reads only the scalar `metrics_per_run.csv`, whereas the voltage boxes
need the per-zone voltage timeseries (npz).

## Validation
- `py_compile` clean.
- Synthetic end-to-end test (8 fake runs, 5 variants x 3 zones): renders 15
  grouped boxes with the V_ref line + legend; graceful skip confirmed on
  old-style npz (no `Vz__` keys).

## Open / next
- **Re-run required to populate:** per-zone voltage is computed at sim time and
  stored in `ts_*.npz`; existing batch npz lack it, and `--replot` cannot
  back-fill. Re-run `experiments/006_CIGRE_MONTECARLO.py --runs 100 --jobs 20`
  (no `--resume`) from `experiments/`.
- Design choice: boxes pool {min,mean,max} per step (faithful "range"); could
  switch to mean-only (cleaner level distribution) or per-bus voltages via
  `record.plant_tn_voltages_pu` if a true per-bus distribution is wanted.
