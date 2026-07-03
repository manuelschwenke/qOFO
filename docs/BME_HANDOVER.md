# BME — Session handover (updated 2026-07-02, Phase 4 core done)

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
| 2 — CommonObjective (+ Convention-A g_own primitives) | ✅ 24 tests green | `1a0881e` |
| 3 — CoordinationBus + signals | ✅ 15 tests green | `dc8cf9b` |
| 4a — Q7 scope + BMEGradientAssembler + HARD GATE identity ✅ (D7 REVISED → complex boundary) | ✅ 15 tests green (BME suite 86) | `f481479` |
| 4b — config / TSOController / runner wiring | ✅ | `81d3e23`, `d71d030` |
| 4c — validation runs: mode="none"/vref BITWISE == pre-BME baseline; bme smoke runs end-to-end; slack-actuator correction | ✅ | see git log 2026-07-03 |
| 5 — Discrete hygiene (notices, slotting, ε-acceptance, ledger) | ✅ 14 tests (sweep 139; bitwise re-verified) | see git log 2026-07-03 |
| 6 — Evaluation ladder + Monte Carlo | 🚧 6a ✅ (w_Φ = 1e5) + 6b ✅ (D6: ε = 5.2e3, c = 1.0e3/5.2e3) + 6c ✅ (D2: edges (1.01, 1.05), w_band = 1e4 — Manuel's centred corridor; uniform-Φ-metric fix; per-zone Φ_i recorded) — next: oracle rung (d), metrics, MC. Calibration sweeps: 120-min horizon (Manuel) | see BME_STATUS.md Phase 6 |

**READ FIRST for Phase 6:** `BME_STATUS.md` Phase 6 §6a (w_Φ calibration
outcome + the three OOS robustness fixes) and Phase 5 section (gate
design, carve-outs) + spec §5 Phase 6 / §6. The ladder rungs (a) none
(b) vref (c) bme (+ bme_loss ablation) (d) oracle share ONE scenario
definition (`experiments/011_BME_LADDER.py`), only the coordination
config differs. Work items: (1) **w_Φ calibration ✅** — resolved via the
`bme_gradient_scale` scalar (NOT gw_precondition reshaping; see §6a for
the rationale), chosen 1e5, filled into `011_BME_LADDER.py`; over-drive
edge at ~1e7 (V escape + solver stress). (2) D6 calibration (ε_switch ≈
5× median per-step |ΔΦ̂| from a baseline run — in the w_Φ-SCALED units;
c_switch per device class); (3) w_band + soft-edge sweep (D2) incl. the
w_band=0 ablation rung — CAVEAT: with w_band=0 the default
`g_z_voltage=1e-12` is inert and voltages escape at hot w_Φ (measured);
decide `zone_g_z_voltage` for the ablation rung with Manuel; consider the
(w_Φ × w_band) pairing jointly; (4) oracle rung (d) from the V5/central
controller machinery with Φ as objective (D8); (5) metrics module (gap
to oracle, Phulpin fairness, oscillation indicator — Φ trajectory,
switch counts, band-violation already in `011_BME_LADDER.py`); (6) MC
campaign (load scenarios × d ∈ {0,1,2,5} × H error × β × ε_switch,
parquet + summary md; ledger is the §3.10.2 premise data — reachable via
the `pre_loop_hook` state dict). Closed-loop counter-switch scenario
(Phase 5 deferral) belongs here. bme rung config MUST set:
`local_sensitivities_*=False`, `refresh_shared_jac_on_tso=True`,
`tso_g_q_tie=0`, `shunt_dispatch != "integrator"` (carve-out raises).
OOS robustness note: machine outages + refreshed Jacobians are handled
now (`actuator_active` masking; `_ppc_bus_is_internal` guards in
`jacobian.py` compute_dQgen_*) — disconnected actuators contribute
exactly-zero columns/rows, u-alignment preserved.

**READ FIRST for Phase 5:** `BME_STATUS.md` Phase 4 sections (esp. the
slack-actuator correction and the smoke-config requirements) + spec §3.8
/ §5 Phase 5. Reuse targets from Phase 0: the DSO shunt feedforward
pattern (runner l. ~3302–3376: atomic commit + `q_itf_sh_offset` +
`ShuntDisturbanceMessage` + SMW masking) is the template for §3.8.1
switch-notice consumption; `SwitchNotice` + bus delay/drop already exist
(Phase 3); MIQP-integer vs shunt-integrator scoping per Q5 (ε-acceptance
= MIQP integers only; integrator banks emit notices + ledger entries).
Open calibration items for Phase 6: `gw_precondition` rescaling of the
bme rung (risk #1 — the smoke confirmed Φ-scale moves are near-zero
against the g_v=1e7-tuned G_w), D6 (ε_switch, c_switch), w_band + edges.

**READ FIRST for 4b:** `BME_STATUS.md` Phase 4 section — Q7 outcome
(TS-level scope, `vn_kv_min`), the D7 REVISION (boundary coordinates are
now STACKED ``[Vm_b | θ_b]`` ∈ R^{2|B|}: `mu_stacked()`,
`h_b_stacked()`, bus `n_boundary = 2|B|` — forced by the money test,
diagnostic recorded), and the "Remaining for Phase 4" wiring list.
`controller/bme_gradient.py::BMEGradientAssembler` is the per-zone
object the controller wires in; `tests/test_bme_gradient_identity.py`
shows the complete per-step chain (machinery build → μ publish →
receiver → g_bme) in `_distributed_gradients()`.

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

## What Phase 3 built (15 tests green)

- `core/coordination_bus.py`: `MarginalSignal` / `SwitchNotice` (frozen,
  validated, read-only vectors), `CoordinationBus` (delay d, drop
  probability with drops drawn AT PUBLISH TIME from a seeded bus-owned
  RNG — deterministic regardless of query order; duplicate publishes
  raise; never self-delivers; `drop_log`), `MarginalReceiver` (per-sender
  β low-pass, consecutive-step enforcement, `ReceivedMarginals` with
  `mu_neighbour_sum`) with the §3.8 policies explicit: cold start exactly
  d steps, missing-signal RAISES when drops disabled, hold-last-FILTERED
  under drops, `extended_cold` zero contribution if a sender's first
  signal was dropped, filter initialised by the first sample. Exported
  from `core/__init__.py`. Tests `tests/test_coordination_bus.py` (pure
  numpy, fast).
- The SELF-marginal never touches bus or filter — Phase 4 adds μ_i
  locally (Convention A). Receiver returns the NEIGHBOUR sum only.
- Expected senders default to all other zones (H_{b,i} spans all of B).

## Phase 4 (next): Controller integration — spec §5 tasks + tests

The core phase; hard gate = distributed-equals-centralised gradient test.

- **Config**: map spec §4's `coordination:` block onto `MultiTSOConfig`
  (flat fields, per-zone override dicts idiom — `configs/multi_tso_config.py`):
  `coordination_mode: none|vref|bme` (default "none"), `bme_delay_steps=1`,
  `bme_drop_probability=0.0`, `bme_beta_filter=0.3`, `bme_seed`,
  `bme_w_band` (+ band edges; w_loss=1 fixed by D2). No parallel config
  system. `mode="vref"` = gate the EXISTING `enable_tie_coordination`
  path unchanged; make the two mutually exclusive (fail-fast).
- **mode="bme" objective switch** (D2/Q1/Q3): TSO layer objective becomes
  Φ_i — g_v schedule tracking OFF, `g_q_tie` forced 0 (raise if explicitly
  non-zero), reserve terms off; hard/soft V output constraints stay local
  and unchanged; DSO cascade untouched.
- **Gradient assembly** (Convention A, §0.2 revision): per step
  g_i^bme = g_own + H_{b,i}ᵀ·(μ_i + mu_neighbour_sum), where
  g_own is assembled from `ZoneGradients.d_*` primitives following the
  controller's u-column order `[Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]`
  (per-DER expansion via the existing `_expand_H_to_der_level` E matrix);
  H_{b,i} from `RestrictedSensitivityProvider.view(zone)`; μ_i from
  `ZoneGradients.mu()` (undelayed, unfiltered); neighbour sum from
  `MarginalReceiver.update(k)` (cold start → price term = H_{b,i}ᵀ·μ_i
  only, logged).
- **Per-step sequence** (spec §5): measure → rebuild ZoneGradients at the
  refreshed operating point (MarginalComputer + ZoneGradients are built
  per Jacobian instance — rebuild when the runner refreshes sensitivities)
  → compute μ_i → publish → receiver.update → assemble g → solve MIQP →
  apply. Slotting/ε-acceptance are Phase 5 — leave hooks.
- **G_w/α rescaling** (risk #1, the v1 price failure mode): Φ is in MW —
  orders of magnitude below the g_v=1e7-scale private objective. Use
  `controller/gw_precondition.py` for the bme experiment configs.
- **Tests** (spec §5 Phase 4): (i) `mode="none"` trajectory identical to
  pre-BME baseline; (ii) single-area identity bme == none; (iii) HARD
  GATE: stacked (g_1,g_2,g_3) with d=0, β=1 equals the FD gradient of
  global Φ w.r.t. stacked u at ≥10 randomised points — the Convention-A
  split dΦ/du_i = ∂Φ_i/∂u_i|_{v_b} + H_{b,i}ᵀ·Σ_all μ_j; (iv) vref
  regression.
- **Before wiring the runner: resolve Q7 with Manuel** (band/loss set on
  runner nets includes DN feeder buses below the PCCs — include-all vs
  voltage-level restriction).

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
