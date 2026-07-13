# 2026-06-30 - Mutual-gradient notebook contingency extension

**Timestamp:** 2026-06-30 09:35:03 +02:00
**Scope:** Extended experiments/008_TIE_MUTUAL_GRADIENT_DEMO.ipynb so contingency scenarios can be defined directly in the notebook.

## Change

- Imported ContingencyEvent in the notebook setup cell.
- Added explicit scenario inputs: SCENARIO_NAME, ZONE_VSET, LOAD_STEP_BUS, LOAD_STEP_MVAR, LOAD_STEP_MINUTE, and CONTINGENCY_SPECS.
- Set the default scenario to same zonal voltage references {1: 1.05, 2: 1.05, 3: 1.05} with a 400 Mvar reactive load connected at bus 9 in zone 2 at minute 20.
- Added build_contingencies() so every run receives fresh ContingencyEvent objects; this avoids reuse of mutated load-event indices.
- Added describe_contingencies() and wrote scenario / contingency metadata into the CSV summary rows.
- Made SWEEP_TAG scenario-derived so new contingency runs use separate CSV and figure filenames.
- Added contingency-time markers to the diagnostic voltage and tie-flow plots.

## Method / Structure

The notebook still uses tie007._base_config(live=False) and the existing run_multi_tso_dso runner. The extension only changes notebook-level configuration injection: after setting cfg.zone_v_setpoints_pu, make_config() now assigns cfg.contingencies = build_contingencies() before enabling or disabling horizontal coordination.

## Reason

This enables testing whether the symmetric TSO-TSO mutual-gradient voltage-boundary coordination also helps under coherent zone voltage references when a local reactive stress is introduced by a large 400 Mvar load step in one zone.

## Risks / Notes

- The 400 Mvar bus-9 load-step case is a local voltage-stress scenario, not automatically a proof of ancillary rescue capability.
- Prior 007 diagnostics suggested the current symmetric gradient-exchange mechanism may improve joint voltage/tie-flow metrics without preferentially rescuing the stressed zone.
- Rerun the notebook setup and function cells after changing CONTINGENCY_SPECS; displayed old cell outputs may otherwise remain stale until execution.
