# 2026-10-06: OFO form of the SVR loops (variant O0) and the Case-1 campaign driver

**Timestamp.** 2026-10-06, 17:30–18:30 CEST (Europe/Berlin).

**Reason.** The thesis gets a new short Case 1 (`ch:case_svr`, Chapter 9): "OFO in place of pilot-node secondary voltage regulation". Its first claim is analytical: the integral RVR and the integral RPR are each one OFO step with one input and one tracked output, identical under the match conditions

    g_w,RVR / w_p = T_RVR s_p^2 / T_TS        g_w,RPR / w_Q = T_RPR h_i^2 / dt

Variant **O0** puts this into the closed loop. The design record is in the dissertation repo, `00_daily_log/2026-10-06_part3_case_design.md`.

**Facts, assumptions, scope.**
- **Controlled outputs.** O0 tracks the pilot voltage per area at the RVR and the machine Q at the RPRs. Its soft output constraints are the monitored-bus voltages of the zone (`zone_def.v_bus_indices`, limits `v_min_pu/v_max_pu` = 0.9/1.1) and the machine capability.
- **Actuators.** The same as S1: the reactive level, mapped to the AVR set-points and TS-DER Q by the alignment law; machine taps held; no shunts.
- **Model.** Frozen calibration only, as for S1. The controller never sees the plant.

---

## Changes

| File | Change |
|---|---|
| `controller/svr_controller.py` | New `SVRZoneConfig` fields `rvr_law` / `rpr_law` (`"pi"` default, `"ofo"`), `ofo_w_p`, `ofo_w_q`, `ofo_v_min_pu`, `ofo_v_max_pu`, `ofo_g_z_v`, `ofo_g_z_q`. New function `_ofo_scalar_step` (see Key method). New methods `_ofo_rvr_step` and `_gen_q_limits`. `calibrate()` additionally records `dv_bus_dq` (monitored buses × devices) from the SAME perturbation solves; no extra power flow. `step()` branches on the laws. `SVROutput` carries the active sets. Constructor rejects `"ofo"` with `k_p > 0` (the OFO step has no proportional path), `rpr_law="ofo"` with the voltage-priority guard, and unknown laws. **Classical behaviour is unchanged.** |
| `configs/config.py` | `MultiTSOConfig.svr_rvr_law`, `svr_rpr_law` (default `"pi"`). |
| `experiments/runners/multi_tso_dso.py` | Passes the laws to the SVR. Gives the OFO form the TS-OFO's own `g_v` (tracking weight), voltage limits, `g_z_voltage` and `g_z_q_gen`, so that O0 differs from the TS-OFO only in objective and structure. Records the active sets. |
| `experiments/helpers/records.py` | `zone_svr_ofo_rvr_n_active`, `zone_svr_ofo_rvr_box_active`, `zone_svr_ofo_rpr_active` (defaulted; old pickles still load). |
| `tests/test_svr_controller.py` | 9 new tests (see Verification). |
| `experiments/case1_svr/run_case1.py` | **New** campaign driver. Variants L2, S1m (S1⁻: integral, no guard), S1 (PI 0.2/0.2 + guard), O0, O1, O2 and seven Pareto points. Config = `make_config()` + DSO_3 ×2 + paramset `ch9_selected` + horizon, with live plots off. A STEP_ALLOWED guard is applied to the resolved configs. Writes provenance (git HEAD, status, diff) into the run dir. |
| `experiments/case1_svr/report_case1.py` | **New** metrics. Per area: e_v, pilot e_vp, worst bus, time outside the band, AVR travel, machine-Q movement, taps, O0 active shares. Also post-event growth per event and the S1m-vs-O0 equivalence (all steps and O0-inactive steps). Writes `metrics.csv`, `epost.csv`, `pareto.csv`, `equivalence.json`. |

---

## Key method

`_ofo_scalar_step(w0, lo, hi, g, r, a, b, g_z)` minimises

    g (w − w0)² + g_z Σ [max(0, a − r w)² + max(0, r w − b)²]   over lo ≤ w ≤ hi.

This is `eq:ofo_qp` with one input and the slacks eliminated: for a quadratic slack penalty the optimal slack is the violation itself.
- The objective is convex and piecewise quadratic, and the derivative is monotone, so the minimiser is found by bisection to machine precision.
- If the box-clipped unconstrained step violates nothing, it is returned unchanged. This makes O0 ≡ S1⁻ exact, not approximate.
- The RVR rows are `dv_bus_dq @ headroom`, taken along the headroom branch the level is heading for, and anchored at the MEASURED bus voltages.

**Loop quantity for the PI form (S1).** On a settled scalar plant with gain error γ:

    e(k+1) = (1 − γ(b + K_p)) e(k) + γ K_p e(k−1),   b = T_TS / T_RVR

This contracts while γ(b + 2K_p) < 2 and γK_p < 1. So b + 2K_p plays the role of λ_max(B):
- deployed S1 (300 s, K_p = 0.2): 1.0;
- 198 s: 1.31, matched to `ch9_selected`;
- 150 s: 1.6, the budget ceiling.

---

## Verification

- `pytest tests/test_svr_controller.py`: **30 passed** (21 existing + 9 new). The new tests show:
  - the OFO-RVR step equals the integral RVR (level 0.65), and the OFO-RPR step equals the integral RPR, both to 1e-12;
  - a binding monitored-bus limit stops the level at the limit (w ≈ 0.125 instead of 0.45);
  - a binding machine capability pulls Q to 200 Mvar (the integral law: 206.7);
  - the forbidden combinations raise;
  - the scalar solver returns the unconstrained step exactly.
- `assert_case1_controlled()` passes: every Case-1 step changes only its own mechanism.
- Smoke test (40 min, S1m vs O0) and campaign: see the follow-up entry.

---

## Open / not done

- Nothing is committed (no instruction to commit).
- The runs execute from a local copy of the repo at `%LOCALAPPDATA%\qOFO_runs\qOFO_2027_case1` (the network drive is ~3× slower). Results are copied back to `results/THESIS_case1_svr/`.
- The reference run starts 2016-01-05 08:00, the start of design window W1. It serves as the equivalence check and an illustration, not as validation; the S1/O2 statistics come from the Case-2 Monte-Carlo ensemble.
