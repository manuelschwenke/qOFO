# DSO operating band 0.92–1.08 pu; the weight selection procedure applied again on it

Timestamp: 2026-10-25, ~22:00 Europe/Berlin.

Reason: author decision. With the selected weights of 2026-10-25 one STS
reached 1.096 pu; the W1 quick test (`2026-10-25d_...`) showed an operating
band of 0.92–1.08 pu in every DSO controller keeps the weak STSs inside it
with e_v(TS) unchanged. The band is configuration ("the DSOs keep a strategic
margin of 0.02 pu"), and the procedure is applied again on it so the weights
are selected with it in place.

## What changed

| File | Change |
|---|---|
| `configs/config.py` | `dso_v_min_pu = 0.90`, `dso_v_max_pu = 1.10` (defaults = the controller's own, so earlier runs are reproduced); `__post_init__` refuses an inverted or non-positive band. |
| `experiments/runners/multi_tso_dso.py` | The band is passed to every `DSOControllerConfig` (the runner never passed one before); the DSO live plot draws it. |
| `experiments/run_multi_system_ofo.py` | `make_config()` sets **0.92 / 1.08**. Only this hunk is committed; the author's uncommitted horizon/contingency edit in the same file stays uncommitted. |
| `experiments/ch_9_parameter_selection/ch_9_3_stage2_screen.py` | `--windows` subset to simulate (the analytic screen still uses all six conditions); recorded in `run_meta.json`; unknown names refused. |
| `tests/test_dso_voltage_band.py` | New, 6 tests: default equals the controller default; invalid bands refused; `make_config` sets 0.92/1.08; a controller build (no time step) gives every DSO controller the configured band. |
| `docs/tuning/2026-10-25_weight_selection_procedure.md` | New, frozen: the 2026-10-24 rules unchanged; configuration adds the band; Stage II simulates W1–W4, W6 (W5 enters neither filter nor ranking; it is simulated for the result only); 38 workers for II-a, 30 afterwards (author: computing power needed tomorrow); result `ch9_selected_band_2026_10`. |
| `docs/tuning/2026-10-24_weight_selection_procedure.md` | Status note: superseded; text and record unchanged. |

`experiments/runners/cascade.py` (single-system runner, own config class) is
not changed.

## Verification

- `tests/test_dso_voltage_band.py` 6 passed; `tests/tuning/test_parameters.py`
  and `tests/test_dso_v_relief_pairing.py` pass.
- Driver (`results/THESIS_ch8_weight_selection_band_20261025/driver.py`):
  configuration check reads band (0.92, 1.08) and banks `miqp`; Stage II-a dry
  run with `--windows W1,W2,W3,W4,W6`: 119 / 512 survivors, as before (the
  band does not enter the screen).
- Full suite not re-run (it would compete with the campaign).
