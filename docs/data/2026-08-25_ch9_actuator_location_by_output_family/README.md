# Ch. 9.1 experiment A — worst settling per output family (V and Q)

Source run: `results/THESIS_ch9_1_timescale_settling/actuator_location_sweep_t0/20260822-014209`
(70/70 cases complete, 2026-08-22), re-aggregated 2026-08-25.

| File | What it is |
|---|---|
| `summary_by_family.md` | worst nodal voltage and worst interface Q, side by side — the Ch. 9.1 table |
| `results_by_family.csv` | per case: `t_settle_s` (V), `t_settle_q_s` (Q), `t_settle_dso_q_s`, `t_settle_tso_q_s`, `q_minus_v_s` |
| `summary.md`, `results.csv` | the **unmodified** originals of the source run |
| `run_meta.json` | provenance of the source run |

No RMS was re-run. `signal_metrics.csv` already carried all 6090 per-output
rows (70 cases x 87 outputs: 70 nodal voltages, 12 DSO interface-Q flows,
5 TSO boundary-Q flows), each with its own band, settling time, excursion and
censoring flag. The case-level worst is a pure reduction over those rows, so
reducing them per family re-reads the same measurements. `signal_metrics.csv`
itself (1.2 MB) stays in `results/` and is not copied here.

**Both families are reported, against their own bands (1 mpu, 1 Mvar), and are
never pooled into a single number.** Voltage and interface Q are not
commensurable, and a single `max` over both hides which family bound the
result — over this run that max was a Q row in 53 of 70 cases, so a lone
column would read as a voltage claim and would not be one.

Supersedes `docs/data/2026-08-24_ch9_actuator_location_voltage_only/`, which
reported the voltage worst with the pooled max beside it.
