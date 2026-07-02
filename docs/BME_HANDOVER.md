# BME — Session handover (updated 2026-07-02, end of Phase 2)

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
| 2 — CommonObjective (+ Convention-A g_own primitives) | ✅ 24 tests green | see git log 2026-07-02 |
| 3 — CoordinationBus + signals | ❌ next | — |

All spec DECISIONS are resolved with Manuel (2026-07-02) — do NOT re-ask;
read `BME_STATUS.md` §0.7. The ones that shape all remaining code:

- **Gradient architecture: Convention A** (REVISED 2026-07-02 after Manuel's
  locality clarification — supersedes an earlier Convention-B recording; see
  `BME_STATUS.md` §0.2 revision note). Under `mode="bme"`:
  `g_i^own = ∂Φ_i/∂u_i` with boundary voltages HELD FIXED, computed from the
  zone-internal port-frozen Jacobian (same J_int factorisation as
  `MarginalComputer` — local by construction), and the price term sums
  **J = ALL zones including i itself** (self-marginal μ_i undelayed and
  unfiltered; neighbour μ_j delayed d, filtered β). Identity to verify:
  `dΦ/du_i = ∂Φ_i/∂u_i|_{v_b fixed} + H_{b,i}ᵀ·Σ_{all j} μ_j`.
  The audit FINDING that the *existing private* gradient assembly is
  Convention B (default shared-Jacobian mode) still stands — it applies to
  `mode="none"`, not to the BME Φ-gradient.
- **Only supra-local object: the H_{b,i} slice** (∂v_b/∂u_i, boundary rows ×
  own actuator columns), served by `RestrictedSensitivityProvider` from the
  plant's global Jacobian purely as the simulation stand-in for what each TSO
  would identify from its own actuator moves + boundary measurements.
  Everything else (Φ_i, μ_i, g_own) is area-local by construction.
- `mode="bme"` does NOT restrict `local_sensitivities_tso` — each zone's OWN
  loop (output constraints, prediction) may keep its Ward Jacobian; that only
  carries the pre-existing experiment-004 model-quality trade-off, orthogonal
  to the Φ-gradient identity.
- Under `mode="bme"`: TSO objective becomes Φ_i = w_loss·P_loss + φ_band
  (w_loss = 1); **g_v schedule tracking OFF**; `g_q_tie` forced 0 (fail-fast);
  hard/soft voltage output constraints stay local and UNCHANGED; DSO cascade
  untouched. w_band calibrated in Phase 6; keep a w_band = 0 (losses-only)
  ablation rung. G_w/α rescaling via `controller/gw_precondition.py` will be
  needed (the old v1 price design died on exactly this scale mismatch).
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

## What Phase 2 built (all tests green)

- `controller/common_objective.py` — `CommonObjective` (Φ_i values from res
  tables, D1 ownership + 50/50 ties, φ_band C¹ hinge, `phi_global()`
  invariant oracle; w_band is REQUIRED, no default — 0.0 is the ablation
  rung) and `ZoneGradients` (`mu()`, plus Convention-A g_own primitives
  `d_q_injection` / `d_pcc_set` / `d_vgen` / `d_tap_2w` / `d_shunt`).
- `sensitivity/marginal_computer.py` extended: `mu_x()` (full-state μ
  chaining, angles + aux stars), `frozen_input_response()` and
  `response_to_q_injection/vgen/tap_2w` (port-frozen Convention-A
  operators; ownership-enforced; hv-side taps only).
- `sensitivity/index_helper.py`: `get_ppc_line_index()`.
- Tests `tests/test_common_objective.py` (24): partition invariant (21 OPs ×
  2 rungs, ≤1e-9), μ FD at every adjacent boundary bus ≤2 % (far endpoints
  isolate the direct tie terms), hinge edges, g_own FD (Q ≤5 %, V_gen ≤5 %,
  taps/shunt whole-step ≤15 %), locality/fail-fast.
- GOTCHA resolved: `_compute_dg_dtau_2w` ALREADY existed in jacobian.py
  (returns a 3-tuple incl. dQ_direct) — `jacobian.py` is net-unchanged.
- **Q7 raised (open, ask Manuel before Phase 4 runner integration):**
  N_i^own = `zone_buses()` closure ⇒ on runner nets the band (and losses)
  would include DN feeder buses under the PCC couplers. Include-all is
  implemented per the spec's literal reading; a voltage-level restriction
  is a config decision, not to be made silently.

## Phase 3 (next): CoordinationBus + signals — spec §5 tasks + tests

Design notes:

- Dataclasses `MarginalSignal` / `SwitchNotice` (spec §4 shapes; frozen,
  registry-order μ vector + v_b snapshot). Natural home: a new module
  beside `core/message.py` (which holds the vertical `CapabilityMessage` /
  `ShuntDisturbanceMessage` patterns) — e.g. `core/coordination_bus.py`.
- `CoordinationBus`: in-process pub/sub, integer delay d (publish at k →
  visible at k+d, NOT earlier), optional drop probability (own
  `np.random.default_rng(seed)`, deterministic), no hidden global state.
- Receiver-side low-pass μ^filt = (1−β)·μ^filt + β·μ (β = 0.3, D3) lives
  with the RECEIVER (per §3.4) — self-marginal bypasses bus AND filter
  (Convention A, d = 0/β = 1 for the self term).
- Cold start (§3.8): first d steps explicitly uncoordinated + logged;
  after warm-up a missing expected signal RAISES unless drop simulation is
  enabled → hold-last-filtered-value, logged per occurrence.
- Today all horizontal exchange is same-step direct method calls in the
  runner (l. 2882–2955) — the bus with d ≥ 1 is genuinely new; keep it
  runner-agnostic (zones interact with bus + plant only, §3.9).
- Tests (spec §5 Phase 3): delay semantics, cold-start behaviour,
  missing-signal raise, hold-last-value under drops, fixed-seed
  determinism.

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
