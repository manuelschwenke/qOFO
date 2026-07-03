# 2026-07-02 — BME Phase 4 core: Q7 scope, D7 revision (complex boundary), gradient assembler, HARD GATE passed

**Context:** BME build, Phase 4 of the spec (`docs/BME_SPEC.md` §5).
Continues Phase 3 (commit `dc8cf9b`, same day). This entry covers the
Phase 4 CORE (the §3.5 money test); config/controller/runner wiring is
Phase 4b (next session).

## Q7 resolved (Manuel): Φ scope = transmission level only

- `CommonObjective` gains `vn_kv_min` (default 0.0 = include-all):
  band only on buses with vn_kv ≥ threshold; a branch contributes losses
  only if EVERY terminal is in scope. IEEE 39 @ 220 kV: 345 kV lines +
  the two 345/345 kV interconnectors; machine trafos, generator
  terminals and DN feeders excluded. Ownership (D1) unaffected.
- Gradient side weight-driven: excluded branches carry weight 0; an
  excluded machine trafo's tap stays an actuator (indirect 345 kV
  response only; its explicit ∂P_ℓ/∂τ term is weighted to zero).
- Tests: TS-scope partition invariant, μ FD, V_gen/tap FD, manual value
  semantics (`tests/test_common_objective.py` → 29 green).

## D7 REVISED: boundary coordinates are complex (forced by the money test)

- First run of §3.5 test 2 FAILED at 17 % on a zone-1 DER column.
  Diagnostic by elimination (scratchpad, numbers recorded in
  `BME_STATUS.md`): single-area TOTAL analytic gradient == FD to
  0.001 % (all Phase 2 pieces exact); the residual is exactly the
  boundary-ANGLE channel (dθ_b/du × dΦ/dθ_b) that the D7
  magnitudes-only design truncates. A loss objective is strongly
  angle-coupled — the truncation is structural, not numerical.
- Resolution per D7's pre-authorised fallback (complex boundary
  quantities, Manuel 2026-07-02): stacked coordinates
  ``[Vm_b (registry) | θ_b (registry)]`` ∈ R^{2|B|} throughout the
  exchange path:
  - `MarginalComputer`: θ-port columns; `mu_x_stacked()`; V-only APIs
    unchanged (Phase 1/2 oracles keep their meaning); reference-bus
    port raises; single-area degenerate mode (portless zone allowed IFF
    the topology has no boundary at all).
  - `RestrictedSensitivityProvider`: `h_b_stacked()` from full state
    responses; magnitude rows cross-checked ≡ legacy `h_b` (1e-9);
    `ZoneBoundaryView.h_b_stacked()` = second permitted read (same §3.9
    informational scope).
  - `ZoneGradients`: `mu_stacked()`; θ direct terms = loss
    angle-gradient at adjacent buses (φ_band has no θ channel).
  - Bus usage: `n_boundary = 2|B|`; `v_b_meas` snapshot stacked.
- Chapter note: θ_b observability = PMU at boundary substations — an
  explicit assumption to record (tie-in: PMU stub of the loss-objective
  work).

## New: `controller/bme_gradient.py`

`BMEGradientAssembler` — per-zone Convention-A assembly:
`g_own()` over `[Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]` from the
Phase 2 primitives; `mu()` = stacked μ_i; `g_bme(mu_total)` = g_own +
h_b_stackedᵀ·μ_total (μ_total = local self term + receiver neighbour
sum). `pcc_hv_buses()` mirrors provider/controller PCC resolution.

## HARD GATE PASSED — `tests/test_bme_gradient_identity.py` (15 green)

- §3.5 test 2: stacked distributed gradients == FD of global Φ w.r.t.
  stacked u; neighbour μ through the REAL bus/receiver (d = 0, β = 1);
  full column coverage at base + 2 OPs; 10 further randomised OPs
  (reduced columns). Continuous ≤5 %, whole-step discrete ≤15 %;
  objective = Q7 TS scope with active tight band.
- §3.5 test 1: single-area identity (no boundary → μ empty, price 0,
  frozen == total) against FD.
- Full BME suite after the revision: **86 passed**.

## Remaining (Phase 4b)

Config fields, TSOController mode switch (Φ replaces g_v tracking,
g_q_tie forced 0), runner per-step wiring, gw_precondition rescaling,
mode="none"/vref trajectory regressions — starting points in
`docs/BME_HANDOVER.md`.
