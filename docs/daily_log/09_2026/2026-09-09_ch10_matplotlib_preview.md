# 2026-09-09 - Run-local Matplotlib preview for Ch 10

Timestamp: 2026-09-09T12:52:02+02:00 (Europe/Berlin).

## Change and reason

Added `experiments/ch_10_case_study/ch_10_2_preview.py` for initial inspection
of saved variant-ladder results without exporting into the dissertation.
The existing `ch_10_2_figures.py` already uses Matplotlib, but its default
destination is the dissertation; `ch_10_3_export_tikz_data.py` is separate.

The new script reads saved pickles and writes PNG/PDF figures, a voltage
metrics CSV, and provenance/metric notes under the selected run's `figures`.
It supports an explicit run directory, newest numbered run by default,
variant filtering, interface-variant selection, output formats, and DPI.
Direct file execution and module execution are supported; rendering uses Agg
and disables external TeX. Existing experiment and dissertation exporters
are unchanged.

## Method and interpretation

- Three shared-axis subplots show recorded `zone_v_rms_err_pu` for TS areas
  1, 2, and 3, overlaying the selected ladder variants.
- An aggregate plot uses `sqrt((e_v,1^2 + e_v,2^2 + e_v,3^2)/3)` with equal
  area weights. This is not a bus-count-weighted whole-system RMS.
- Interface tracking uses saved actual Q and dispatched Q references per
  transformer, grouped by DSO network (O3 by default).
- A further three-panel figure shows each area's voltage minima and maxima.
- The CSV distinguishes sample mean, sample RMS, and peak of e_v. The
  sample RMS matches the ladder report's calculation, whose printed title
  currently calls it a time mean. No existing metric implementation changed.
- Missing area errors stay missing; the aggregate requires all three areas.
  Empty/missing variants are reported and skipped; invalid timestamp order
  is rejected and different variant time grids are reported.

These are offline post-PF plant outcomes, not the noisy pre-control signals
available to controllers. No plant/controller access or architecture changes.
Actuators remain the saved variant's AVR/DER, OLTC, and shunt configurations;
controlled outputs inspected are TS bus voltage and TS-DS interface Q.
The run folder contains no saved configuration/event schedule, so no assumed
voltage bounds or contingency markers are drawn. Single-run figures do not
support statistical confidence statements.

## Usage and validation

```powershell
python -m experiments.ch_10_case_study.ch_10_2_preview --run-dir results/ch10_ladder/0002
```

Run 0002 has seven nonempty variants (L1, L2, S1, O1, O2, O3, M), each with
360 records at 20-second spacing, from 20 seconds through 120 minutes.
Validation: executed in qOFO_clean against run 0002; generated four PNGs, four PDFs, 28 complete CSV metric rows, and preview notes. Confirmed aggregate equivalence with the existing voltage_rms_err_all helper for O3, checked RMS against an analytic example, and checked that a missing area produces an aggregate gap. Inspected all four rendered layouts. The initial shared-y plot clipped later-area peaks; fixed by applying the lower bound only after plotting all areas, then regenerated and rechecked. No simulations were rerun.

Suggested Obsidian follow-up: link the run's `figures` directory from the
Chapter 10 single-run comparison note, and retain the explicit distinction
between spatial RMS, its sample mean, and its sample RMS.
