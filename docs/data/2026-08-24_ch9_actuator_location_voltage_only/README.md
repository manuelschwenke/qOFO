# Ch. 9.1 experiment A — worst settling reduced over nodal voltages only

Source run: `results/timescale/actuator_location_sweep_t0/20260822-014209`
(70/70 cases complete, 2026-08-22), re-aggregated 2026-08-24.

| File | What it is |
|---|---|
| `summary_voltage_only.md` | the voltage-only reduction and its delta against the original |
| `results_voltage_only.csv` | per case: both reductions side by side, plus `delta_vs_all_s` |
| `summary.md`, `results.csv` | the **unmodified** originals, reduced over every output |
| `run_meta.json` | provenance of the source run |

No RMS was re-run. `signal_metrics.csv` already carried all 6090 per-output
rows (70 cases x 87 outputs: 70 nodal voltages, 12 DSO interface-Q flows,
5 TSO boundary-Q flows), each with its own band, settling time, excursion and
censoring flag. The case-level worst is a pure reduction over those rows, so
restricting it to `output_kind == "voltage"` re-reads the same measurements.
`signal_metrics.csv` itself (1.2 MB) stays in `results/` and is not copied here.

**Scope of the resulting claim.** The bound covers the settling of the nodal
voltage outputs only. Interface Q is a controlled output of both layers, so its
settling is not covered; `t_settle_all_s` is the column that still carries it.
