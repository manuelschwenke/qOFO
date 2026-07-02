# 2026-07-02 — BME Phase 2: CommonObjective (Φ, φ_band, μ, Convention-A g_own)

**Context:** BME build, Phase 2 of the spec (`docs/BME_SPEC.md` §5; status
`docs/BME_STATUS.md`). Continues Phase 1 (commit `d2d7f1c`). All DECISIONS
D1–D8 resolved 2026-07-02; gradient architecture = Convention A (§0.2
revision note).

## What was changed

### New: `controller/common_objective.py`

- `CommonObjective` — the common coordination objective Φ = Σ_i Φ_i
  (spec §3.3):
  - Φ_i = w_loss·P_i^loss + Σ_{n ∈ N_i^own} φ_band(v_n); losses are the
    ACTUAL res-table branch losses over owned branches (D1: interior
    branches by endpoint owner, tie lines split 50/50 via
    `BoundaryTopology.tie_loss_shares()`); N_i^own =
    `topology.zone_buses(i)` (closure ownership).
  - φ_band: C¹ quadratic hinge, value + gradient; edges default ±3 % (D2);
    **w_band is a required argument** (no silent default — magnitude is a
    Phase 6 calibration item; 0.0 = losses-only ablation rung).
  - `phi_global(net)` — computed WITHOUT the ownership map (all in-service
    branches/buses): the independent oracle for the partition invariant.
  - Fail-fast: missing power-flow results, NaN res entries, cross-zone
    non-tie branches, missing owned buses/ties all raise.
- `ZoneGradients` (via `CommonObjective.gradients(marginal_computer)`) —
  area-local gradient bundle at the cached Jacobian operating point:
  - State-space loss gradient dP^loss/d(θ, V): analytic weighted row-sums
    of the MATPOWER dSbr_dV identities over the ppc `Yf`/`Yt` matrices
    (formulas in the module header), in MW matching the res tables.
  - `mu()` — μ_i = dΦ_i/dv_b (§3.4): interior gradient chained through
    `MarginalComputer.mu_x()` + direct terms at adjacent boundary buses
    (loss sensitivity at own ports AND far tie endpoints; band gradient at
    own ports only, D1).
  - Convention-A own-gradient primitives (§3.5): `d_q_injection`,
    `d_pcc_set` (negated, load convention), `d_vgen` (incl. explicit
    pinned-terminal loss + band terms — the terminal magnitude IS the
    input), `d_tap_2w` (incl. closed-form ∂P^loss_ℓ/∂τ of the trafo's own
    branch), `d_shunt` (−q_step·V² scaling, mirrors
    `compute_dV_dQ_shunt`).

### Extended: `sensitivity/marginal_computer.py`

- `mu_x(grad_x_int, grad_direct)` — μ assembly from a gradient over the
  FULL internal state (θ and V, aux 3W star states) — needed because the
  loss gradient is angle-dependent; `mu()` now delegates to it (Phase 1
  behaviour unchanged, 26 tests re-run green).
- Port-frozen input responses (Convention A): `frozen_input_response(dg)`
  = −J_int⁻¹·∂g_int/∂u (J_int retained from the Phase 1 build; dropping
  the port rows IS the "boundary voltages held fixed" operation), plus
  `response_to_q_injection` / `response_to_vgen` / `response_to_tap_2w`.
  Ownership enforced (§3.9 locality: foreign actuators raise); PV-bus
  injections, the slack machine and non-hv tap sides raise.
- New public `sens` / `topology` properties.

### Extended: `sensitivity/index_helper.py`

- `get_ppc_line_index()` — pandapower line → ppc branch row (mirrors
  `get_ppc_trafo_index`; positional, non-contiguous indices safe).

### `controller/__init__.py`

- Exports `CommonObjective`, `PhiBreakdown`, `ZoneGradients`.

### NOT changed

- `sensitivity/jacobian.py` is net-unchanged: a duplicate
  `_compute_dg_dtau_2w` briefly added during the session was removed once
  the EXISTING helper (l. 2696, tie-coordination era, returns
  `(dg_dtau, Δτ, dQ_direct)`) was found — it is reused as-is (`dQ_direct`
  is a Q-observation correction, not needed for Φ whose explicit
  τ-dependence enters via the branch-loss term).

## Tests

`tests/test_common_objective.py` — **24 passed** (full BME suite: 60; plus
controller-side regressions `test_tso_output_gradient`,
`test_tie_coordinator*`, `test_tso_loss_objective`: 32 passed):

- Partition invariant Σ_i Φ_i == Φ_global: base + 20 randomised operating
  points (loads ±10 %, gen ±0.01 pu, seed 123), ≤1e-9 relative, for the
  losses-only AND active-tight-band rungs; Φ_global (w_band = 0) equals the
  raw res-table totals to 1e-12; zone 1's 50/50 tie split recomputed by
  hand.
- μ FD (money test): every adjacent boundary bus of every zone, ≤2 %, both
  rungs — extended Phase 1 oracle (zone closure + far tie endpoints, ALL
  adjacent buses pinned by voltage sources). Far endpoints isolate the
  direct tie-share terms. Sparsity: exact zeros outside the adjacent set.
- φ_band hinge: zero inside/at edges, quadratic outside, C¹ gradient,
  one-sided curvature 2·w_band vs 0.
- g_own FD with ports pinned: Q injection ≤5 % (2 buses/zone), V_gen ≤5 %
  (1 gen/zone), OLTC whole-step secant ≤15 % (4 trafos incl. trafo 0 whose
  hv bus is a boundary port), synthetic 20 Mvar shunt ≤15 %.
- Locality/conventions: foreign actuator raises, PV-bus injection raises,
  slack machine excluded, PCC negation exact, topology-mismatch raises.

## Open point raised (Q7 — for Manuel, before Phase 4 runner integration)

`zone_buses()` closure ownership means that on RUNNER nets (with DSO
feeders attached) Φ_i's band and losses would include DN buses/branches
under the PCC couplers. Implemented per the spec's literal N_i^own
(include-all); restricting the band set (e.g. by voltage level) is a
config decision that must not be made silently. Recorded in
`BME_STATUS.md` Phase 2 findings.

## Why

Spec §5 Phase 2 tasks + §0.2 revision consequence (ii): CommonObjective
must expose the frozen-boundary own-gradient pieces so Phase 4 can
assemble g_i^bme = g_i^own + H_{b,i}ᵀ·Σ_j μ_j^filt and pass the
distributed-equals-centralised identity test.
