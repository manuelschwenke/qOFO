# Corrected Chapter 9 Experiments B1 and B2

Use these runs as the final Chapter 9 timescale evidence:

- `B1_ninner_10mvar/20260821-180548/` ? 120-case isolated
  (N_\mathrm{inner}) experiment;
- `B2_ts_period_sweep/20260821-183927/` ? 60-run
  (T_\mathrm{TS}) sweep.

Both runs use the post-selection DSO-4 safety correction recorded in each
`run_meta.json` and in `_ch9_selected_design_snapshot.py`:

- DSO-2/4 voltage relief factor: 10, with the OLTC weight paired so
  (g_v/g_{w,\mathrm{OLTC}}) is preserved;
- DSO-4 continuous-DER weight: (828.3682036634343);
- DSO 1?3 global continuous-DER weight: (549.881810527467).

B1 headline: 120/120 converged, median 2, p90 6.2, p95 10, maximum 18,
113/120 at or below 9, and zero tap moves or voltage violations.

B2 headline: 60/60 runs completed. Median tracking ratio decreases from 0.4407
at 60 s to 0.1909 at 300 s; censoring is 15.5?16.6%. Do not use B2's `n_k`
as the dissertation (N_\mathrm{inner}), and do not infer a unique 180 s
optimum without a supervisory staleness/objective metric.

Full interpretation:
`../../daily_log/08_2026/2026-08-21_dso4_damping_b1_b2_corrected.md`.

Dissertation-edit prompt:
`../../handoff/2026-08-21_ch9_b1_b2_corrected_dissertation_prompt.md`.

