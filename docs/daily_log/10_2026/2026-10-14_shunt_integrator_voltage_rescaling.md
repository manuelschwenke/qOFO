# Shunt integrator: per-iteration voltage rescaling of the bank sensitivities

- Timestamp: 2026-10-14 (CEST).
- Reason: Author decision while revising the thesis (Ch 3 `eq:shunt_rescaling`, Ch 4 shunt dispatcher, Ch 5 `eq:shunt_interface_offset`). The thesis states that a bank's sensitivities are rescaled at every iteration by `(V_meas / V_0)²`, because a constant-susceptance bank injects `q_step · V²` per step. In the code that rescaling (`SensitivityUpdater`) acted only on shunt columns of an MIQP's H; with `shunt_dispatch='integrator'` no H has shunt columns, so the integrating dispatcher evaluated the per-step injection at the cached linearisation voltage `V_0` throughout. The code now does what the thesis states.

## What changed

| File | Change |
|---|---|
| `sensitivity/sensitivity_updater.py` | New module-level `voltage_rescaling_ratios(v_measured_sq, v_cached_sq)` (ratio + the existing 1e-10 identity guard), now also used by `SensitivityUpdater._update_shunt_columns` (identical arithmetic). New `bank_voltage_ratio(measurement, cached_net, bus_idx)`: `(V_meas / V_0)²` at one bank bus, `V_0` from `cached_net.res_bus`; warns and returns 1.0 if the bus is not measured. |
| `configs/config.py` | New `MultiTSOConfig.shunt_int_v_rescale: bool = True`. `False` reproduces runs before 2026-10-14. |
| `experiments/runners/multi_tso_dso.py` | Import of `bank_voltage_ratio`. Integrator block: per bank `r_b = bank_voltage_ratio(meas_i, sens_i.net, bus)`; multiplies the voltage part `h_col` (hence also the overshoot guard's `h_v`), the Q_PCC part `h_qpcc` and the SG-Q (reserve) part of `g_H`. Commit block: the interface feed-forward `dq_itf` is multiplied by `r_c = bank_voltage_ratio(meas_i, dso_ctrl.sensitivities.net, bus)` (DSO model's cached V_0, pre-switch measured tertiary voltage). |
| `tests/test_shunt_voltage_rescaling.py` | New: ratio values + identity guard, inputs not mutated, cached-voltage lookup, missing-bus fallback with warning, and on a real pandapower feeder `h(V_0)·r = h` with the injection at the measured voltage. |

## Key structure

- The TSO measurement (`measure_zone_tso`) carries all bus voltages, so the tertiary bank bus is always measured in the QSS runner.
- Unchanged: integrator state, clamp and hysteresis thresholds stay in nameplate Mvar (`q_step` at 1 pu); the rank-1 Sherman–Morrison update at a commit (`apply_shunt_step_change_smw`) is untouched; the cached operating point is never refreshed.
- Other callers of the `compute_*_shunt` functions (TSO H shunt columns, coordinator cross-sensitivities, DSO shunt block) belong to the legacy `'miqp'` shunt path / `shunt_bus_indices`, which is empty in integrator mode; their rescaling is the existing `SensitivityUpdater`.

## Validation

- `tests/test_shunt_voltage_rescaling.py` 5 passed; `tests/test_sensitivity_updater.py` and `tests/test_shunt_integrator.py` pass.
- `tests/test_shunt_integrator_integration.py`, `tests/test_tso_tertiary_shunt.py`, `tests/test_tso_output_gradient.py` fail for a reason independent of this change (the error is raised in the network builder, before any shunt code runs): they call `build_ieee39_net(scenario="base")`, which is no longer in the scenario registry (valid: `base_410`, `rural_700`; `network/ieee39/build.py` last changed 2026-08-13). Not fixed here.
- Smoke run (scratch script, `ch9_selected` via `build_selected_config`, 15 min, `shunt_int_g_w=1e-4` and dwell 0 to force commits; NOT a representative operating case): finished without error, 45 records in 368 s; 86 `bank_voltage_ratio` calls on 6 distinct cached models (2 TSO + 4 DSO, i.e. gradient and feed-forward paths both exercised); 0 missing-bus warnings; 7 commits in both directions (`dQ_itf` 19–27 Mvar for 25 Mvar banks). Ratios ranged 0.858–1.175, i.e. tertiary voltages up to about −7 %/+8 % from the linearisation point after forced switching -- larger than the "3 %" example below, because engaged banks move their own tertiary voltage away from V_0 (where all banks were off).
- Note: running the runner with stdout redirected on Windows needs `PYTHONIOENCODING=utf-8` (the commit print contains "ℓ").

## Follow-up (same day): outdated tests repaired

- Author decision: only scenario `rural_700` is needed for now (not `base_410`).
- Five test files used the removed scenario `"base"`: `test_shunt_integrator_integration.py`, `test_tso_tertiary_shunt.py`, `test_tso_output_gradient.py`, `test_msc_msr_banks.py`, `test_tso_loss_objective.py`. Changed to `scenario="rural_700"`.
- Second cause after that change: the `rural_700` build-time state does not solve single-slack (documented in `network/ieee39/hv_networks.py::_reinit_pf`, 2026-07-30). The tests' `pp.runpp(net, run_control=False, calculate_voltage_angles=True)` calls now pass `distributed_slack=True`, as the runner does (`MultiTSOConfig.distributed_slack = True`). The small local test network in `test_tso_tertiary_shunt.py` (l. 65, plain `pp.runpp(net)`) is unchanged.
- Result: the five files, 27 passed (were 20 failed / 7 passed with only the scenario fixed, all failing tests on the first `pp.runpp`).

## W1 comparison, rescaling on vs off (author-requested cheap check)

- Setup: `experiments.case1_svr.run_case1.base_config(180)` (= `make_config()`: full cascade TS-OFO + STS-OFO, shunt integrator, start 2016-01-05 08:00 = W1, its contingency schedule; DSO_3 ×2; paramset `ch9_selected`), 180 min; the two legs differ only in `shunt_int_v_rescale`. Both ok, 540 records, ~620 s wall each. Scratch scripts (session scratchpad, not in the repo): `run_variant.py`, `compare.py`.
- Bank commits: on 9, off 8. Same sequence, except (a) the second MSC step at bus 113 (zone 3) at 53.7 min (on) vs 56.7 min (off), i.e. one TS period earlier, and (b) one extra MSR step at bus 53 (DSO_1 tertiary) at the final instant 179.7 min (on). Feed-forward magnitudes `dQ_itf` differ by up to 2.3 Mvar per step (e.g. 28.14 vs 25.91, 25.85 vs 23.50).
- TS voltages: mean RMS deviation from V_set per zone 4.039/6.973/5.475 mpu (on) vs 4.037/6.959/5.495 mpu (off); voltage ranges identical up to 0.2 mpu; largest instantaneous difference of the zone-mean voltage 1.2/2.0/4.6 mpu (zone 3 during the 3-min shift of (a)). TS OLTC steps 7 vs 7. HV voltage ranges within 1.1 mpu.
- Interface tracking RMS [Mvar], whole horizon: DSO_1 2.24 vs 1.69, DSO_2 0.86 vs 0.86, DSO_3 3.00 vs 3.04, DSO_4 1.77 vs 1.88. The DSO_1 increase is entirely the final record (13.1 Mvar transient at t = 180 min after commit (b)); excluding the last minute: DSO_1 1.49 vs 1.55, DSO_4 1.73 vs 1.84, i.e. slightly better with the rescaling.
- Reading: in this window the effect is small: bank timing shifts by at most one TS period, voltage metrics are unchanged at the 0.02 mpu level, and interface tracking improves marginally. One window, one realisation; not a statistical statement.

## Consequences / open

- **All Part III results produced before this change ran with the cached V_0** (equivalent to `shunt_int_v_rescale=False`): Ch 8 weight selection and C/D experiments, Ch 10 ladder/MC, Ch 11. Author decision (2026-10-14): **no re-run for now**; a W1 comparison (180 min, on vs off) is run to size the effect; the expected effect is small (per-step injection off by about 2·ΔV/V_0, e.g. ~6 % for a 3 % tertiary voltage deviation, and corrected by feedback at the next iteration), but it changes when banks commit.
- Not committed to git (the working tree carries other uncommitted work).
