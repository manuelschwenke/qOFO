# OFO tap lock: 5 iterations, wall-clock lock off

- Timestamp: 2026-10-14 (CEST).
- Reason: Author decision while revising thesis Ch 3 ("Wear and switching constraints" of the OLTC subsection). The thesis now states one tap lock for both layers, 5 iterations of the owning controller: 100 s at T_STS = 20 s and 15 min at T_TS = 180 s. The second, wall-clock lock of the OFO path is deactivated.

## What changed

| File | Change |
|---|---|
| `controller/base_controller.py` | `OFOParameters.int_cooldown` default 6 → 5 (docstring added). `int_cooldown_s` default unchanged (0). |
| `configs/config.py` | `MultiTSOConfig.int_cooldown` 6 → 5. New field `ofo_oltc_cooldown_s: float = 0.0` (wall-clock OLTC lock of the OFO MIQP path; 0 = off). `oltc_cooldown_s` docstring: now applies to local-mode tap relays only. |
| `configs/cascade_config.py` | `int_cooldown` 6 → 5. |
| `experiments/runners/multi_tso_dso.py` | The three `OFOParameters(...)` constructions (TSO, DSO, central) pass `int_cooldown_s=config.ofo_oltc_cooldown_s` instead of `config.oltc_cooldown_s`; comment at the local-mode rate limiter updated. |
| `experiments/ch_9_parameter_selection/ch_9_1_ts_period_sweep.py` | Docstring of the lockout inference updated (logic unchanged: the iteration lock dominates). |

## Key structure

- Lock semantics unchanged: after a switch at iteration k the actuator is pinned (`u_lower = u_upper = u_current`) while `iteration < k + 1 + int_cooldown`, so the next 5 iterations are locked (`base_controller.py`, Step 3c / Step 11).
- The wall-clock machinery (`int_cooldown_s`, Step 3d) is kept and tested; it is only no longer fed by the runner. Decoupling via a new field instead of setting `oltc_cooldown_s = 0` keeps the local-mode `DiscreteTapControl` limiter (`oltc_cooldown_s_mt` = 180 s, `_nc` = 60 s, fallback 30 s) unchanged for the local-control baselines.
- `int_cooldown` also applies to in-MIQP shunts where present; in the thesis configurations shunts are dispatched by the shunt integrator (own dwell `shunt_int_t_dwell_s` = 30 min), so the change affects OLTCs only.

## Validation

- `tests/test_oltc_cooldown.py`: 9 passed.
- Defaults checked: `MultiTSOConfig.int_cooldown = 5`, `ofo_oltc_cooldown_s = 0.0`, `oltc_cooldown_s = 30.0`; `OFOParameters.int_cooldown = 5`, `int_cooldown_s = 0.0`.
- No test asserts the old value 6.

## Consequences / open

- **Part III results produced before this change ran with 6 iterations + 30 s wall clock** (Case 1, Ch 8 C/D1/D2, Ch 10 ladder/MC). Author decision (2026-10-14): Ch 8 is **not** re-run; the results are expected to be similar under the 5-iteration lock. Recorded here so the difference stays traceable.

## Follow-up (same day): tuning harness aligned

- `tuning/parameters.py` `FIXED_OVERRIDES["int_cooldown"]` 1 → 5 (author decision), so future tuning campaigns run with the same lock as the evaluated runs. Campaigns before 2026-10-14, including the one that selected 399e30d3b60b, ran with 1. Import check: `FIXED_OVERRIDES` gives `int_cooldown = 5`, `int_max_step = 1`; no test references the value.
- Shunt dwell confirmed for the thesis text: `shunt_int_t_dwell_s = 30 min` (`configs/config.py`), passed per bank to `ShuntIntegratorConfig.t_dwell_s` (`multi_tso_dso.py` ~3289), checked as `t_now - last_switch_t >= t_dwell_s` (`controller/shunt_integrator.py:235`); not overridden by `make_config` or `ch9_selected` (its `hunt_dwell_h` is a selection limit, not the dwell).
