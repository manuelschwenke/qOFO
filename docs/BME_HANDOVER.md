# BME — Session handover (written 2026-07-02, end of Phase 1)

For the next Claude Code session continuing the BME build. Read in this order:

1. `docs/BME_SPEC.md` — the driving spec (verbatim; phases §5, binding conventions §8).
2. `docs/BME_STATUS.md` — living status: component map, audits A1–A3, ALL
   decision outcomes (D1–D8, Q1–Q5), Phase 0/1 sections, architectural
   placement (§0.11).
3. This file — practical state + Phase 2 starting points.

## Where we are

| Phase | State | Commit |
|---|---|---|
| 0 — Reconnaissance + audits + decisions | ✅ | `426293e` |
| 1 — BoundaryTopology / RestrictedSensitivityProvider / MarginalComputer | ✅ 26 tests green | `d2d7f1c` |
| 2 — CommonObjective | ❌ next | — |

All spec DECISIONS are resolved with Manuel (2026-07-02) — do NOT re-ask;
read `BME_STATUS.md` §0.7. The ones that shape all remaining code:

- **Convention B** (audit A1): zones' own gradients are total derivatives via
  the shared full-network Jacobian ⇒ the BME price term uses
  **J = neighbours only** (self-marginal would double-count).
- Under `mode="bme"`: TSO objective becomes Φ_i = w_loss·P_loss + φ_band
  (w_loss = 1); **g_v schedule tracking OFF**; `g_q_tie` forced 0 (fail-fast);
  hard/soft voltage output constraints stay local and UNCHANGED; DSO cascade
  untouched. w_band calibrated in Phase 6; keep a w_band = 0 (losses-only)
  ablation rung. G_w/α rescaling via `controller/gw_precondition.py` will be
  needed (the old v1 price design died on exactly this scale mismatch).
- `mode="bme"` REFUSES `local_sensitivities_tso=True` (Ward-PQ mode is
  neither Convention A nor B).
- `mode="vref"` = wrap the EXISTING `controller/tie_coordinator.py` path
  unchanged (§3.7 hypothesis NOT confirmed — separate paths).
- Discrete hygiene: round robin slotting; ε-acceptance applies to MIQP
  integers only (MSC/MSR integrator banks keep their own commit rule but emit
  notices + ledger entries); d = 1 default, d = 0 supported; β = 0.3.

## What Phase 1 built (all committed, tests green)

- `network/boundary_topology.py` — `BoundaryTopology`: registry B (ascending
  bus order — IEEE39 3-area: [1, 2, 8, 13, 14, 16, 17, 26, 38]), B_ij, ties
  (pp lines 2, 14, 25, 5, 18), per-zone `own_boundary` / `adjacent_boundary`
  / `interior_buses` / `zone_buses` (closure ownership), `tie_loss_shares()`
  (50/50), hard separator assertion.
- `sensitivity/boundary_sensitivity.py` — `ZoneInputSpec` (columns
  `[Q_DER(bus-level) | Q_PCC_set | V_gen | s_OLTC | s_shunt]`, mirrors
  `TSOController._build_sensitivity_matrix` conventions),
  `RestrictedSensitivityProvider.h_b(zone)` / `.view(zone)`
  (`PermissionError` out-of-scope). Per-DER expansion stays controller-side
  (`_expand_H_to_der_level`).
- `sensitivity/marginal_computer.py` — `MarginalComputer`: ports = zone's own
  boundary buses; `response_v()` = ∂v_int/∂v_b; `response_full()` = full
  (θ, V) state response **ready for the Phase 2 loss gradient**;
  `mu(grad_v_int, grad_direct)` → length-|B| vector, exact zeros outside
  `adjacent`, direct terms outside adjacent raise.
- Tests: `tests/test_boundary_topology.py`, `tests/test_boundary_sensitivity.py`,
  `tests/test_marginal_computer.py`. The μ FD oracle pattern (reuse it in
  Phase 2!): `pandapower.toolbox.select_subnet` over `zone_buses(z)` + one
  `create_ext_grid` per port at the plant operating point → reproduces
  interior voltages < 1e-6 pu, then central-difference the port vm.

## Phase 2 (next): CommonObjective — spec §5 tasks + tests

Design notes prepared this session:

- Φ_i losses: use ACTUAL branch losses over owned branches (pandapower
  res tables: `res_line.pl_mw`, `res_trafo.pl_mw`, `res_trafo3w.pl_mw`) for
  the value / partition invariant; ties weighted by `tie_loss_shares()`.
  Ownership of branches: both endpoints owned by one zone ⇒ that zone
  (guaranteed by the separator); tie lines 50/50.
  The controller's form-B loss (`_loss_line_coeffs`, c_ℓ·|I_ℓ|² over
  monitored lines) is a SURROGATE — fine for the controller gradient, but
  the invariant test Σ Φ_i == Φ_global must use actual losses (open point
  Q4 in `BME_STATUS.md` §0.6: coverage of ALL owned branches).
- Loss gradients dP_loss/d(θ, V) are analytic from branch admittances and
  endpoint complex voltages; chain through `MarginalComputer.response_full()`
  for the μ loss part, and through the zone H rows for the control-space
  part (Convention B).
- φ_band hinge: value + one-sided gradients at the edges get their own unit
  tests (spec Phase 2); band edges default ±3 % (D2), owned buses =
  `topology.zone_buses(z)`; note PV-bus V is pinned (their band gradient has
  no internal-response channel — document).
- Partition invariant test: ≥ 20 randomised operating points (perturb loads /
  gen vm with fixed seed, re-run PF).

## Practical gotchas (save yourself an hour)

- Python: `C:\Users\Manuel Schwenke\.conda\envs\qOFO_clean\python.exe`;
  run tests as `python -m pytest tests/test_*.py -q` from the repo root.
- The repo is on a network share (`Z:`): recursive Glob/grep over the whole
  tree TIMES OUT — search per directory.
- IEEE39 facts (corrected during Phase 1, tests document them):
  the slack is a `slack=True` GEN at a 10.5 kV terminal bus behind boundary
  bus 38's machine trafo (there is NO ext_grid; bus 38 is PQ, live H/μ rows);
  0-idx bus 19 (IEEE 20) is REMOVED by `build_ieee39_net` (two-trafo-chain
  collapse); the slack machine + its trafo are NOT zone actuators (no
  Jacobian column at the reference bus — exclude from `ZoneInputSpec`).
- IEEE bus 3 (0-idx 2) is a boundary bus of TWO zone pairs (B_12 ∩ B_23) —
  handled by the registry embedding, no special-casing.
- Conventions (spec §8, binding): fail-fast, British English, §3-symbol
  header map in every new module, scoped `git add` + commit per phase with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, daily-log entry in
  `docs/daily_log/` per work package, update `BME_STATUS.md` at phase
  boundaries (record decision outcomes with dates).
- Auto-memory: `tso_tso_tie_coordination.md` in the project memory directory
  carries the same state — update it at the next phase boundary too.
