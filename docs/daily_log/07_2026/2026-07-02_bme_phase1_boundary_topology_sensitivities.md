# BME Phase 1 — boundary topology, restricted sensitivities, marginal computer

**Date:** 2026-07-02
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Phase 1 of the BME build plan (spec §5), after Manuel resolved all
DECISIONS D1–D8 and open questions Q1/Q2/Q3/Q5 (recorded with dates in
`docs/BME_STATUS.md` §0.7). Convention B (audit A1) is the binding gradient
convention: the BME price term will use J = neighbours only.

## What was added

1. **`network/boundary_topology.py`** — `BoundaryTopology` / `TieLine`:
   boundary registry B in fixed ascending order, per-pair subsets B_ij, tie
   orientation `zone_i < zone_j`, per-zone own/adjacent boundary sets,
   closure-based bus ownership (D1; generator-terminal and orphan buses
   inherit their electrical zone), 50/50 tie-loss shares, and the hard
   vertex-separator assertion of spec §3.2 (cross-zone non-line branches and
   spanning components raise with the "enlarge B" message).
2. **`sensitivity/boundary_sensitivity.py`** — `ZoneInputSpec`,
   `RestrictedSensitivityProvider`, `ZoneBoundaryView`: assembles
   H_{b,i} = ∂v_b/∂u_i from the shared full-network Jacobian with the exact
   column conventions of `TSOController._build_sensitivity_matrix`
   (PCC load-convention negation, shunt V² step scaling, V_gen via
   `compute_dV_dVgen_matrix`); the zone-bound view makes §3.9's access
   restriction enforceable (`PermissionError` on out-of-scope reads).
3. **`sensitivity/marginal_computer.py`** — `MarginalComputer`: area-internal
   reduced Jacobian with the zone's own boundary buses as fixed-magnitude
   ports (R = −J_int⁻¹·∂g_int/∂V_port), exposing ∂v_int/∂v_b and the full
   state response for the Phase 2 loss gradient; `mu()` embeds into registry
   order with exactly-zero entries outside the zone's adjacent boundary set.
4. Package exports added to `network/__init__.py` and
   `sensitivity/__init__.py` (only change to existing files).

## Method / key structure

The interior block is extracted in the polar mismatch space
(rows [P_PV, P_PQ, Q_PQ], states [θ, V]) using the existing
`index_helper` mappings; ports contribute either their V-state Jacobian
column (PQ port) or a `_compute_dg_dVgen` voltage-source column (pinned
port). 3W star buses of zone-owned transformers are included in the interior
block. The FD oracle for μ builds a per-zone port sub-network
(`pandapower.toolbox.select_subnet` + one voltage source per port at the
plant operating point) — which first *numerically asserts §3.2's separator
consequence* (interior voltages reproduced to <1e-6 pu), then validates
dΦ/dv_b per port by central differences (≤2 % agreement).

## Verification

26 new tests green (`test_boundary_topology.py` 12,
`test_boundary_sensitivity.py` 8, `test_marginal_computer.py` 6), plus
`test_sensitivity_updater.py` re-run green after the export additions.
FD coverage: V_gen (3 zones), Q_DER, one whole OLTC tap step, one whole
shunt step, μ per zone per port.

## Corrections to Phase 0 records (reason for two BME_STATUS.md edits)

* **Bus 38 (IEEE 39) is NOT a pinned slack bus.** `swap_slack_to_bus38`
  installs a ``slack=True`` gen at a 10.5 kV terminal bus behind a machine
  trafo; bus 38 itself is PQ, so its H_{b,i} row and μ entries are live.
  The Phase 0 "inert rows / inert band penalty" note was wrong and has been
  corrected in §0.4 and D1.
* **Bus 19 (IEEE 20) root cause:** removed by `build_ieee39_net`'s
  two-trafo-chain collapse (`network/ieee39/build.py` l. 249–291) — not a
  tagging issue. Documented by test.
* The slack machine and its trafo are excluded from zone actuator specs
  (no Jacobian column at the reference bus), matching the runner's
  `ZoneDefinition` convention.
