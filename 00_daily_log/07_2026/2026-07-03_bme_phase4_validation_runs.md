# 2026-07-03 — BME Phase 4 closed: validation runs (bitwise regressions, bme smoke, slack-actuator correction)

**Context:** completes Phase 4 (core `f481479`, wiring `81d3e23` +
`d71d030`). Scenario for all runs: 005 CIGRE cascade config shortened to
30 sim-minutes (90 plant steps, 10 TSO ticks), headless, scratchpad
scripts, child processes per repo checkout.

## Results

1. **mode="none" trajectory == pre-BME baseline, BITWISE.** Git worktree
   of `0d7c47b` (last commit before any BME code) vs current HEAD, same
   scenario: 90-step `total_losses_mw` trajectories identical to the
   last bit (max |Δ| = 0.0). Spec §5 Phase 4 regression (i) in its
   strong (solver-deterministic) reading.
2. **vref regression, BITWISE.** `enable_tie_coordination=True` with the
   new `coordination_mode="vref"` alias reproduces the pre-BME tie
   coordination run exactly (max |Δ| = 0.0) — regression (iv).
3. **mode="bme" end-to-end smoke: runs.** First live execution of the
   full chain on the real cascade net (3W PCC couplers, DN feeders — Q7
   scope and 3W machinery exercised): 90 steps, 36.1 s vs 32.5 s
   baseline (~10 % per-tick overhead), cold start logged, no failures.
   Losses end 31.0 vs 30.1 MW (baseline): the MW-scale Φ gradient makes
   near-zero moves against G_w weights tuned for the g_v=1e7 private
   objective — `gw_precondition` rescaling (risk #1) is the Phase 6
   rung-configuration task, not a wiring defect. Config requirements:
   the bme rung must set `local_sensitivities_*=False` and
   `refresh_shared_jac_on_tso=True` (005 defaults violate both; the v1
   validation raises as designed).

## Code changes (driven by the smoke run)

**Phase 1 finding corrected: the slack machine IS a zone actuator.**
The runner's `ZoneDefinition` includes the slack machine's AVR setpoint
(gen at the reference bus 40) and its machine trafo 12 as ordinary
actuators; the first bme tick raised on the old "slack is not an
actuator" fail-fast. BME columns must match the controller's u exactly,
so support was added:

- `sensitivity/marginal_computer.py`: `response_to_vgen` accepts the
  reference bus (the slack magnitude is an exogenous power-flow input;
  ∂g/∂V_ref via `_compute_dg_dVgen` is well-defined — the Phase 1 note
  concerned the missing STATE column, not this input channel). New
  module-level `dg_dtau_2w_tolerant(sens, trafo_idx)` mirroring
  `compute_dV_ds_2w`'s accumulate-only-existing-rows behaviour (the
  existing `_compute_dg_dtau_2w` raises when a terminal is the
  reference); `response_to_tap_2w` uses it.
- `sensitivity/boundary_sensitivity.py`: `_assemble_stacked` OLTC
  columns use the tolerant assembly.
- `tests/test_bme_gradient_identity.py`: zone 1's full-coverage spec now
  carries the slack V_gen column and trafo 12 — both FD-confirmed by the
  hard gate (29 tests green across identity + Phase 1 suites).
- `tests/test_common_objective.py`:
  `test_slack_machine_not_an_actuator` → `test_slack_machine_vgen_supported`
  (pins the revised convention; value FD-pinned by the identity test).

## Status

Phase 4 ✅ (see `BME_STATUS.md`). Next: Phase 5 discrete hygiene
(switch notices via the DSO feedforward pattern, round-robin slotting,
ε-acceptance on MIQP integers, switching ledger).
