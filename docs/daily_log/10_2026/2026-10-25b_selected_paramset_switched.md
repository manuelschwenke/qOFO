# SELECTED_PARAMSET switched to the procedure's result

Timestamp: 2026-10-25, ~14:40 Europe/Berlin.

Reason: author decision to use the result of the single application of
`docs/tuning/2026-10-24_weight_selection_procedure.md`
(`2026-10-25_weight_selection_result.md`).

## What changed

| File | Change |
|---|---|
| `experiments/ch_9_parameter_selection/_ch9_selected_design.py` | `SELECTED_PARAMSET` "ch9_selected_10_2026" → **"ch9_selected_final_2026_10"** (c08b1d3d33ab); `INITIAL_PARAMSET` "stage1_start" → "stage1_start_10_2026_banks" (the procedure's Stage-I point; the old set lacks `g_w_tso_shunt`). Comments record both changes. |

`experiments/run_multi_system_ofo.py` imports `SELECTED_PARAMSET`, so its
default follows without an edit (its comment at l. 340 still names the old
set; the file carries the author's uncommitted horizon/contingency edit and
was left untouched).

## Verification

- `build_selected_config()` → candidate c08b1d3d33ab, `g_w_tso_shunt`
  25.50, banks `miqp`, DSO_3 ×2; `INITIAL_PARAMSET` builds weight-complete;
  `apply_paramset(make_config(), SELECTED_PARAMSET)` passes `require_weights`.
- `tests/tuning/test_parameters.py`, `test_stage1_banks.py`,
  `test_stage2_shunt_wear.py`: 33 passed; `tests/experiments/test_thesis_runs.py`:
  11 passed.

## Not changed

`experiments/helpers/rms_cosim_config.py::DEFAULT_PARAMSET` is still
"ch9_selected" (left open by the author on 2026-10-20); that set lacks
`g_w_tso_shunt`, so the RMS co-simulation needs `--paramset` until it is
switched.
