# 2026-06-03 — Fix Fig. 3a per-zone voltage plot (plot_cigre.py)

**Reason:** `plot_fig3a_voltage(per_zone=True)` raised
`AttributeError: 'MultiTSOIterationRecord' object has no attribute 'zone_v'`.
The records never stored per-bus voltage arrays (`zone_v`); they store the
per-zone *spatial RMS* error in `zone_v_rms_err_pu` (with documented fallback
`|zone_v_mean - v_set|`).

**Changes (`visualisation/plot_cigre.py`):**
- Imported `voltage_rms_err_per_zone` from `experiments.helpers.comparison_metrics`.
- Replaced the inline `_zone_rms_err_trace` (which read the nonexistent
  `r.zone_v` / `r.t_s`) with a precompute step that calls
  `voltage_rms_err_per_zone(logs[name], v_set)` once per variant.
- Zone IDs now derived from the data (union of `pz["zones"]`) rather than the
  hardcoded `[1, 2, 3]`; subplot count follows the number of zones. One subplot
  per TSO zone, as requested.
- Updated docstring to reflect the real data source.

**Key structure:** per-zone series come straight from the record field
`zone_v_rms_err_pu[z]` (p.u.), scaled to mp.u. for the y-axis. Consistent with
how the system-wide path already uses `voltage_rms_err_all`.

**Verified:** `python -m py_compile visualisation/plot_cigre.py` → OK.
