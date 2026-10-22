# 2026-07-03 — BME Phase 4b (part 2): runner per-step wiring

**Context:** BME Phase 4 wiring, continuing part 1 (config + controller
hook, commit `81d3e23`). This entry: the `multi_tso_dso.py` integration.

## What was changed (`experiments/runners/multi_tso_dso.py`)

### Setup block (after the tie-coordinator construction)

- Mode validation: `coordination_mode` ∈ {none, vref, bme};
  "vref" requires `enable_tie_coordination=True` (alias for the
  existing gated path — unchanged, per spec §1); "bme" fail-fasts on:
  `enable_tie_coordination=True` (double steering),
  `numerical_h=True` (BME needs the analytic shared Jacobian),
  `local_sensitivities_*=True` (v1: the runner freezes reduced
  Jacobians; Ward-loop variant = future wiring), and
  `refresh_shared_jac_on_tso=False` (v1 realises §3.4's "μ at the
  measured state of step k" by re-linearising shared_jac every TSO
  tick; measurement-evaluated gradients on a frozen model are noted
  future work).
- Builds once: `BoundaryTopology` from `tn_zone_map`; `CommonObjective`
  from `bme_w_band` / band edges / `bme_vn_kv_min` (Q7); per-zone
  `ZoneInputSpec` from `ZoneDefinition` — DER columns are BUS-level in
  the controller's DERMapping first-seen unique order (raises on
  duplicate DER buses without a mapping); `enable_bme_mode()` on every
  TSO controller (raises if `g_q_tie != 0`, Q3); `CoordinationBus`
  with stacked coordinates (n = 2|B|, D7 revision) and one
  `MarginalReceiver` per zone (β = `bme_beta_filter`, start_step 0).

### Per-TSO-tick round (inside `run_tso`, after measurements, before the zones solve)

Sequence per spec §5 Phase 4: rebuild
`RestrictedSensitivityProvider` + `MarginalComputer` + `ZoneGradients` +
`BMEGradientAssembler` at the freshly re-linearised `shared_jac`
operating point → compute μ_i (stacked ``[dΦ/dVm_b | dΦ/dθ_b]``) →
publish (`v_b_meas` = stacked [vm | θ_rad] snapshot; step index =
`tso_step_count − 1`, a consecutive TSO-tick counter) →
`receiver.update` → μ_total = μ_i (local, undelayed, unfiltered) +
filtered delayed neighbour sum (cold start: self term only — the
receiver logs the §3.8 events) → `receive_bme_gradient(g_bme)` one-shot
injection into each zone controller; the controller's
`_compute_objective_gradient` then returns the per-DER-expanded g_i^bme.

## Tests

Suites re-run green (56): `test_tso_output_gradient`,
`test_tie_coordinator`, `test_tie_coordination_hooks`,
`test_bme_gradient_identity` (hard gate), `test_coordination_bus`.
Runner/config import smoke passed. NOTE: no pytest exercises
`run_multi_tso_dso` itself — the end-to-end validation runs are the
explicit remaining Phase 4 item (see `BME_STATUS.md`): mode="bme" smoke
(expect G_w/α calibration via `gw_precondition`, risk #1), mode="none"
trajectory comparison vs the pre-BME baseline, vref regression run.

## Why

Spec §5 Phase 4 per-step sequence. The runner mediates all horizontal
exchange (zones interact with bus + plant only, §3.9), mirroring how the
tie-coordination round is embedded; `coordination_mode="none"` leaves
the loop byte-for-byte untouched (inert branch checks only).
