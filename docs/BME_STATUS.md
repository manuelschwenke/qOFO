# BME — Boundary Marginal Exchange: Status File

Driving spec: *Boundary Marginal Exchange (BME) — Multi-TSO Coordination for OFO-MIQP*
(design specification and build plan, received 2026-07-02; saved verbatim as
`docs/BME_SPEC.md`). Session handover for continuation sessions:
`docs/BME_HANDOVER.md`.
This file is the living status record required by spec §5; one section per phase,
updated at phase boundaries. All DECISION outcomes (§7 of the spec) are recorded
here with dates.

| Phase | Title | Status |
|---|---|---|
| 0 | Repository reconnaissance | ✅ 2026-07-02 (closed: DECISIONS D1–D8 and Q1/Q2/Q3/Q5 resolved with Manuel, see §0.7) |
| 1 | Boundary topology and sensitivities | ✅ 2026-07-02 (26 tests green; see Phase 1 section) |
| 2 | Common objective | ✅ 2026-07-02 (24 tests green; see Phase 2 section) |
| 3 | Coordination bus and signals | ✅ 2026-07-02 (15 tests green; see Phase 3 section) |
| 4 | Controller integration | ✅ 2026-07-03 — hard gate PASSED (incl. slack columns); runner wired; mode="none" and vref BITWISE identical to the pre-BME baseline; bme end-to-end smoke runs (see Phase 4 section) |
| 5 | Discrete hygiene | ✅ 2026-07-03 (14 tests green; slotting + ε-acceptance + ledger + notices wired; two documented carve-outs — see Phase 5 section) |
| 6 | Evaluation ladder + Monte Carlo | ❌ not started |
| 7 | Analysis artefacts | ❌ not started |

---

## Phase 0 — Repository reconnaissance ✅ (2026-07-02)

Read-only. No source files modified (this status file and a daily-log entry are
the only files created). One read-only scratch diagnostic (boundary enumeration)
was executed from the session scratchpad, outside the repository.

### 0.1 Component-mapping table (spec §2 abstract names → repository)

| Spec component | Repository location | Notes |
|---|---|---|
| Multi-area simulation module | `experiments/runners/multi_tso_dso.py::run_multi_tso_dso` (l. 293); zone bookkeeping in `controller/multi_tso_coordinator.py` (`ZoneDefinition`, `MultiTSOCoordinator`) | Per-step orchestration; `MultiTSOCoordinator` also computes cross-zone blocks H_ij and the preconditioned system matrix M_sys with the contraction criterion (l. 19–37) — directly reusable for §3.10.1 |
| Per-area TSO controller + per-step MIQP | `controller/tso_controller.py::TSOController`; solve sequence `controller/base_controller.py::BaseOFOController.step` (l. 441–664); problem build `optimisation/miqp_solver.py::build_miqp_problem` | u = [Q_DER \| Q_PCC_set \| V_gen \| s_OLTC \| s_shunt], y = [V \| Q_PCC \| I \| Q_gen \| Q_tie]; α scales continuous moves only, integers move whole steps and are rounded (base l. 559–619) |
| Sensitivity provider | `sensitivity/jacobian.py::JacobianSensitivities` (+ `build_sensitivity_matrix_H`); one **shared full-network** instance assigned to every controller by default (`multi_tso_dso.py` l. 1828, 1947–1950); optional per-zone Ward reduction `sensitivity/network_reduction.py`; rank-1 SMW refresh `sensitivity/sensitivity_updater.py`; finite-difference H `sensitivity/numerical_h.py` | H_{b,i} row/col selection of the global H is available by construction in the default mode |
| CAIR (capability restriction) | `core/message.py::CapabilityMessage`; DSO side `controller/dso_controller.py::generate_capability_message` (l. 459); TSO side `controller/tso_controller.py::receive_capability` (l. 637) → `_compute_input_bounds` (l. 1293) | DSO capability intervals become box bounds on Q_PCC_set in U_i(k); untouched by BME |
| Hysteresis quantiser + dwell logic | MIQP integer handling in `base_controller.py`: ±`int_max_step` per step (l. 512–517), iteration cooldown `int_cooldown` (l. 519–523), wall-clock OLTC cooldown `int_cooldown_s` (l. 525–538, 625–641); MSC/MSR banks: `controller/shunt_integrator.py` (hysteresis half-width `delta`, `t_dwell_s`, daily switching budget) | Two distinct discrete paths: MIQP-internal integers and the separate integrating shunt dispatcher (`_shunt_mode="integrator"`) |
| DSO feedforward correction (MSR/MSC switching) | Runner `multi_tso_dso.py` l. 3302–3360: atomic commit = plant toggle + persistent interface-Q feedforward offset `q_itf_sh_offset` added to setpoint messages (l. 3366–3376) + `ShuntDisturbanceMessage` → `dso_controller.py::receive_disturbance_message` (l. 396) + rank-1 SMW refresh of both cached Jacobians (l. 3330–3336) | This is the pattern §3.8.1 generalises horizontally: measurement-interpretation correction AND estimator (cached-Jacobian) masking both exist |
| TSO–TSO v_ref scheme | `controller/tie_coordinator.py::HorizontalTieCoordinator`; controller hooks `tso_controller.py::receive_tie_coordination` (l. 673), `report_tie_boundary_voltage` (l. 729), `report_boundary_gradient` (l. 867); runner round `multi_tso_dso.py` l. 2882–2955; gate `MultiTSOConfig.enable_tie_coordination` (l. 1004) | See audit A2 — current implementation is the 2026-06-28 *gradient-exchange* redesign, not the plain ΔV_ref relaxation |
| Config system | `configs/multi_tso_config.py::MultiTSOConfig` (dataclass, flat fields + per-zone override dicts); per-experiment `make_cigre_config()` (`005` l. 115, `006` l. 228); vertical cascade `configs/cascade_config.py` | The `coordination:` block of spec §4 must become `MultiTSOConfig` fields (no parallel config system) |
| V1–V5 experiment runner | `experiments/005_CIGRE_MULTI.py` (ladder defined l. 12–18: V1/V2 local, V3 one-sided OFO, V4 cascade, V5 central upper bound), `experiments/006_CIGRE_MONTECARLO.py`; both call `run_multi_tso_dso` | Must remain untouched. V5-style central mode (`controller/central_controller.py`) is a natural starting point for the BME oracle rung (d) |
| Test layout | Flat `tests/` folder: pytest `test_*.py` plus scratch `diag_*.py`; relevant: `test_tie_coordinator.py`, `test_tie_coordination_hooks.py`, `test_tso_output_gradient.py` (gradient invariant), `test_tso_loss_objective.py`, `test_shunt_integrator*.py` | BME tests follow this convention |
| Plant protocol | No plant abstraction class: the plant is the combined pandapower net; measurements via `measure_zone_tso` / `core/measurement.py`, application + PF in the runner (`core/pf_adapter.py`) per `docs/ARCHITECTURE.md` (combined net = plant, model nets = sensitivities only) | Spec's "PandapowerStaticPlant / PowerFactoryPlant" names do not exist; the two-network principle is equivalent and binding |
| Common-objective building blocks | Loss (form B): `tso_controller.py::_loss_line_coeffs` / `_loss_output_grad_i` (l. 1529–1630), weight `g_loss`, P_loss = Σ_ℓ c_ℓ·\|I_ℓ\|² over monitored lines, reuses I-rows of H (daily log 2026-06-30, `test_tso_loss_objective.py`); voltage bands: currently hard/soft **output constraints** with g_z slack in the MIQP, not an objective hinge | Φ_i loss term is largely reusable; φ_band as a C¹ objective hinge is new (Phase 2) |

### 0.2 Audit A1 — gradient convention: **Convention B** (default mode)

**Implemented formula** (`tso_controller.py::_compute_objective_gradient`, l. 1749–1996):

```
grad_f = Σ_terms (∂y_own/∂u_i)ᵀ · ∇_y f_own  +  direct control-variable terms
       = 2·g_v·(V−V^set)ᵀ·∂V/∂u + 2·g_q_tso·(Q_PCC−Q_PCC^set)ᵀ·∂Q_PCC/∂u
         + 2·g_q_tie·(Q_tie−Q_tie^set)ᵀ·∂Q_tie/∂u + reserve terms (SG via ∂Q_gen/∂u,
         DER direct) + (2·g_loss·c_ℓ·|I_ℓ|)ᵀ·∂I/∂u
```

with the invariant `grad_f == ∇_y f · H` pinned by `tests/test_tso_output_gradient.py`
(single output-space source `_compute_output_gradient`, l. 1631–1747).

**Provenance of H decides the convention.** In the default mode
(`local_sensitivities_tso=False`), every zone controller's `sensitivities` is the
**shared full-network** `JacobianSensitivities(net)` (`multi_tso_dso.py` l. 1828,
1947–1950). Each zone's H (`_build_sensitivity_matrix`, l. 2042) is therefore a
row/column selection of the global network response: rows = the zone's own
monitored outputs, columns = the zone's own actuators, but the derivative runs
**through all network paths including neighbour areas**. That is exactly
`g_i^own = dΦ_i^own/du_i` (total derivative) → **Convention B**. Hence in the BME
augmentation **J = neighbours only, j ≠ i**; adding a self-marginal term would
double-count by exactly μ_i.

**Caveat (important):** with `local_sensitivities_tso=True`
(`multi_tso_config.py` l. 413; `sensitivity/network_reduction.py`), each zone's
Jacobian is built from a Ward-style reduced net whose boundaries are **constant-PQ
loads** (`network_reduction.py` l. 21–30) — the boundary voltage is *not* held
fixed. This mode is therefore **neither Convention A (frozen boundary) nor B
(full network)**; it approximates B with an equivalent-model error. Proposal:
`coordination_mode="bme"` fail-fasts unless the full-network sensitivity path is
active (see Open questions Q2).

**REVISION 2026-07-02 (same day, after Manuel's locality clarification —
supersedes the implementation choice above, not the audit finding):** the audit
finding stands as a fact — the *existing private-objective* gradient assembly
is Convention B in the default mode. But the **BME Φ-gradient under
`mode="bme"` will use Convention A**: `g_i^own = ∂Φ_i/∂u_i` with the boundary
voltages **held fixed**, computed from the zone-internal port-frozen Jacobian
(local by construction — the same J_int factorisation as `MarginalComputer`),
and **J = all zones including i itself** (the self-marginal μ_i enters through
the price term, undelayed and unfiltered since it never crosses a border).
Everything in the BME gradient is then area-local **except the one H_{b,i}
slice** (§3.9 concession, access-wrapped; ≙ what each TSO would identify from
its own input moves + boundary measurements). Consequences: (i) `mode="bme"`
does NOT restrict `local_sensitivities_tso` — the zone's own loop
(output constraints, prediction, Q_PCC handling) may keep its Ward Jacobian,
carrying only the pre-existing experiment-004 model-quality trade-off,
orthogonal to the Φ-gradient identity; (ii) Phase 2's `CommonObjective` must
expose the frozen-boundary own-gradient pieces; (iii) the Phase 4 identity
test 2 uses the Convention-A split
`dΦ/du_i = ∂Φ_i/∂u_i|_{v_b fixed} + H_{b,i}ᵀ·Σ_{all j} μ_j`.

### 0.3 Audit A2 — v_ref hypothesis: **not confirmed as stated; delta documented**

The spec's §3.7 hypothesis (implemented scheme = BME with a quadratic boundary
objective, coordination via a price) does **not** match the code. Per the spec's
fallback branch, `mode="vref"` will wrap the existing implementation unchanged;
the two paths stay separate.

What is actually implemented (third design iteration, 2026-06-28 redesign):

1. **Exchanged signal:** one scalar per tie endpoint,
   `γ = (∇_u J · h_b)/(h_bᵀ h_b)` (`report_boundary_gradient`,
   `tso_controller.py` l. 867–904) — the *full private* objective's control-space
   gradient projected onto the boundary-voltage row h_b = ∂V_b/∂u and normalised
   by ‖h_b‖². In the pure-tracking case this reduces to `γ = 2·g_v·(V_b − V_b^ref)`
   — so the *signal* does have the quadratic-boundary marginal shape of §3.7, but
   it is (i) a normalised scalar per tie, not the vector μ = dΦ/dv_b of §3.4,
   (ii) derived from each zone's **private** objective, not a common Φ.
2. **Actuation channel:** the coordinator (`controller/tie_coordinator.py`)
   descends the combined marginal over a **negotiated setpoint** ΔV_ref with a
   per-zone worsening safeguard, deadbanded subsidiarity anchor and clip; zones
   track the redirected boundary setpoint through their existing quadratic g_v
   term (`receive_tie_coordination`, l. 673–727). **No price term enters any
   zone's gradient** (explicit comments at `tso_controller.py` l. 1744–1746 and
   l. 1993–1995).
3. **History that matters for BME:** the *first* design (2026-06-25) was exactly
   a price-in-gradient scheme and **failed** — inert at λ ≪ g_v, power-flow
   divergent at λ ~ g_v (`docs/tso_tso_tie_coordination_concept.md` §7.1(b)).
   Two further documented flaws of the current gradient-exchange scheme are the
   reason for the pivot to BME:
   - **Incommensurate objective scales** across heterogeneous zones: measured raw
     γ ratios of 12 000–30 000× between zones with different objective types
     (`multi_tso_config.py` l. 1076–1109, `tie_normalize_gradients` docstring);
   - **Sticky-OLTC long-run degradation**: a coordinator transient carries an OLTC
     across a tap threshold; the tap is cooldown-locked and never returns; the
     plant is permanently displaced although the continuous coordination state
     recovers (`docs/daily_log/2026-07-01_tso_tso_long_run_degradation_investigation.md`).

**Positioning consequences (for the chapter):**
- BME's single common objective Φ removes the incommensurability by construction
  (no exchange rate between private goals — the Phulpin-2009 distinction of §3.3).
- BME's discrete hygiene (ε-acceptance + slotting + ledger) is aimed squarely at
  the sticky-OLTC mechanism identified on 2026-07-01.
- The v1 price failure is **not** automatically refuted by BME: BME's price term
  is again linear in the gradient. The structural differences (price derived from
  the *same* common objective every zone optimises; correctness pinned by the
  distributed-equals-centralised gradient test; G_w/α recalibrated for Φ's scale)
  are the hypothesis to be *demonstrated* by the Phase 4 money test and Phase 6/7
  evidence — record as hypothesis, not fact.
- The scalar γ infrastructure (measure → report → coordinate → message → track)
  is a validated skeleton for the per-step horizontal round; BME replaces the
  scalar γ with the vector μ and the setpoint redirect with the price term.

### 0.4 Audit A3 — boundary topology, IEEE 39 3-area

Partition: `network/zone_partition.py::fixed_zone_partition_ieee39` (l. 105–168);
tie detection `get_tie_lines` / `get_zone_tie_lines` (l. 527–606); wired into the
runner at `multi_tso_dso.py` l. 725–806. Every TN bus is assigned to exactly one
zone — there are **no jointly-owned boundary buses**; the natural boundary set B
is the set of tie-line endpoint buses, each owned by its zone.

Enumeration (read-only scratch diagnostic, base `build_ieee39_net()`, 2026-07-02;
0-indexed pandapower bus ids, IEEE = 1-indexed):

| Tie line (pp index) | Endpoint A | Endpoint B | Zone pair |
|---|---|---|---|
| 2 | bus 1 (IEEE 2, z1) | bus 2 (IEEE 3, z2) | 1–2 |
| 14 | bus 38 (IEEE 39, z1) | bus 8 (IEEE 9, z2) | 1–2 |
| 25 | bus 26 (IEEE 27, z1) | bus 16 (IEEE 17, z3) | 1–3 |
| 5 | bus 2 (IEEE 3, z2) | bus 17 (IEEE 18, z3) | 2–3 |
| 18 | bus 13 (IEEE 14, z2) | bus 14 (IEEE 15, z3) | 2–3 |

Boundary registry candidate: B = {1, 2, 8, 13, 14, 16, 17, 26, 38} (0-idx)
= IEEE {2, 3, 9, 14, 15, 17, 18, 27, 39}, |B| = 9.

- **Separator property: holds.** Removing B disconnects the zone interiors
  (checked over lines + trafos; no cross-zone branch other than the 5 lines).
  The formal repo-side check is Phase 1 work.
- **Shared boundary bus:** bus 2 (IEEE 3, zone 2) is an endpoint of **two** ties
  (line 2 → zone 1 and line 5 → zone 3), i.e. B_12 ∩ B_23 ≠ ∅. The v_ref
  setpoint-redirect cannot compose there (known limitation, concept doc §7.4);
  BME's additive price composes naturally — a concrete argument for the design.
- **Slack near the boundary** *(corrected in Phase 1)*: bus 38 (IEEE 39) does
  NOT host a pinned voltage source. `swap_slack_to_bus38` installs a
  ``slack=True`` gen at a **10.5 kV terminal bus behind a machine trafo**;
  bus 38 itself is a PQ bus with a live Jacobian voltage state, so its
  H_{b,i} row and μ entries are ordinary non-zero quantities (strongly
  regulated by the adjacent slack machine, but not pinned). The Phase 0
  claim of "inert rows / inert band penalty at IEEE 39" was wrong.
- **Minor surprise to verify in Phase 1:** 0-idx bus 19 (IEEE 20) is absent from
  every zone bus list (zone 3 has 13 of its 14 nominal buses). It is
  zone-interior either way and does not affect the boundary; root cause (subnet
  tag / voltage level in `case39`) to be confirmed when `BoundaryTopology` is
  built.
- DSO feeders attach in zones 2 and 3 (`network/ieee39/constants.py` l. 245–258)
  — vertical cascade unaffected by B.

### 0.5 Information and communication model (spec §3.9, repo-adapted)

| Quantity | Available to area i? | How (today) | How (BME target) |
|---|---|---|---|
| Own model, measurements, own Φ_i | yes | local (`measure_zone_tso`, own H rows) | unchanged |
| v_b^meas at adjacent boundary buses | yes | boundary endpoints are monitored voltage buses (runner l. 797–803 extends v_bus_indices) | unchanged |
| μ_j from neighbours | n/a | — (only scalar γ per tie, exchanged same-step via direct method calls) | CoordinationBus, delay d |
| SwitchNotice from neighbours | n/a | — (vertical ShuntDisturbanceMessage only) | CoordinationBus, delay d |
| μ_i (self) | n/a | — (`report_boundary_gradient` is the scalar precursor) | local reduced Jacobian (new `MarginalComputer`) |
| H_{b,i} = ∂v_b/∂u_i | **concession** | implicitly available (shared full-net Jacobian) but unenforced | `RestrictedSensitivityProvider` enforcing rows(B) × cols(u_i) |
| Neighbour models, internal measurements, objectives | no | not exchanged | unchanged |

Note: today there is **no message bus, no delay, no drop** — all horizontal
exchange happens as direct method calls inside one runner step (γ collected and
consumed before the zones solve, l. 2882–2955). The CoordinationBus with d ≥ 1
is genuinely new machinery.

### 0.6 Open questions and surprises

1. **Q1 — objective replacement under `mode="bme"`.** Spec §1 says Φ
   "replaces/augments" private objectives. The current TSO objective is dominated
   by g_v = 1e7-scale voltage-schedule tracking. Proposal: under `mode="bme"` the
   TSO-layer objective becomes exactly Φ_i (loss + band), zeroing g_v tracking,
   g_q_tie and reserve terms for the coordinated experiment; DSO cascade
   objectives untouched. Consequence: gradient scale changes by orders of
   magnitude → G_w/α recalibration required (`controller/gw_precondition.py`
   exists for exactly this). **Needs Manuel's confirmation** (interacts with D2).
2. **Q2 — BME × `local_sensitivities_tso`.** The Ward-PQ reduced Jacobian is
   neither Convention A nor B (§0.2). Proposal: `mode="bme"` requires the
   full-network sensitivity path and raises otherwise (fail-fast); the local-net
   variant is future work alongside online estimation of H_{b,i}.
   *(This Phase-0 proposal is SUPERSEDED — see the §0.2 revision note and the
   Q2 resolution in §0.7: Convention A, no restriction on
   `local_sensitivities_tso`.)*
3. **Q3 — `tso_g_q_tie` (default 10.0) under BME.** The private tie-flow tracking
   term steers the same boundary the price term prices; keeping both
   double-steers. Proposal: `mode="bme"` forces g_q_tie = 0 (fail-fast if
   explicitly set non-zero).
4. **Q4 — Φ_i loss coverage.** The existing form-B loss term covers *monitored
   lines* of each zone. Φ's partition invariant needs *all owned branches*
   (lines, trafos, tie-line halves per D1). Phase 2 must extend coverage or
   document the monitored-subset approximation and test the invariant
   accordingly.
5. **Q5 — two discrete pathways.** OLTC integers live inside the MIQP; MSC/MSR
   banks live in the separate shunt integrator (own hysteresis/dwell/budget).
   The ε-acceptance rule (§3.8.3) applies naturally to the MIQP integers; for
   integrator-committed banks the switch notice + ledger apply, but the
   frozen-integer QP comparison does not (the integrator has its own commit
   criterion). Proposal: Phase 5 scopes ε-acceptance to MIQP integers and emits
   notices for both pathways. **Needs Manuel's view.**
6. **Q6 — μ at the slack boundary bus** (IEEE 39): structurally ≈ 0 entries;
   assert-and-document rather than special-case.
7. **Surprise (helpful):** `MultiTSOCoordinator` already assembles H_ij,
   M_TSO,ij and M_sys with the contraction criterion — Phase 7's non-cooperative
   eigenvalue cloud is mostly plumbing, not new maths.
8. **Surprise (helpful):** the DSO feedforward pattern already contains both
   ingredients §3.8.1 needs (measurement-side offset + estimator masking via SMW).

### 0.7 DECISION table (spec §7) — **all resolved with Manuel, 2026-07-02**

| ID | Decision | Outcome (2026-07-02) |
|---|---|---|
| D1 | Ownership convention | ✅ Tie-line losses split 50/50 per tie. Band penalty of each boundary bus owned by the bus's own zone (unambiguous — every bus has exactly one zone). *(Phase 1 correction: bus 38 / IEEE 39 is a live PQ bus — the slack sits behind its machine trafo — so no boundary band penalty is structurally inert.)* |
| D2 | w_loss, w_band, edges | ✅ w_loss = 1. **φ_band retained** after clarification: it is *not* voltage optimisation (no schedule is coordinated; hard/soft V output constraints stay local and unchanged) but a soft security margin that makes a zone's voltage *stress* visible in the exchanged price μ — exactly zero inside the band, so Φ ≈ pure losses in normal operation. w_band magnitude and soft-band edges calibrated in the Phase 6 sweep (spec default ±3 % as starting point); an explicit **w_band = 0 (losses-only) ablation** rung is added to the ladder to honour the "losses are the only common currency" reading. Folded-in Q1 resolution: under `mode="bme"` the TSO-layer objective becomes Φ_i (g_v schedule tracking OFF); gradient rescaling via `gw_precondition`. |
| D3 | Filter β | ✅ 0.3 default (Manuel: no preference); MC-swept in Phase 6. |
| D4 | Delay d | ✅ Default d = 1; d = 0 supported (in-process same-step exchange, required by the identity test anyway); sweep {0, 1, 2, 5}. |
| D5 | Slotting | ✅ Round robin, slot length 1 (clarified for Manuel: token passing = a circulating permission-to-switch token, only the holder commits; round robin = fixed rotation k mod N_A). Token passing stays the documented alternative if Phase 6 shows idle-slot cost. Note: slotting stacks on the existing `int_cooldown`/`int_cooldown_s` locks. |
| D6 | ε_switch, c_switch | ✅ Delegated to assistant: ε_switch ≈ 5× median per-step continuous \|ΔΦ̂\| from a Phase 6 baseline calibration run; per-device-class c_switch consistent with the existing dwell/daily-budget wear reasoning; both swept in Phase 6; rationale documented at calibration time. |
| D7 | Boundary quantity | ✅ Voltage magnitudes only; complex voltages noted as an admissible architectural fallback if the design ever requires it (Manuel). Angle dependence of tie flows remains the documented limitation. |
| D8 | Oracle | ✅ Centralised per-step OFO-MIQP (reuse the V5/central controller machinery with Φ as objective). |

Open questions resolved the same day:
- **Q1** → folded into D2 (Φ replaces g_v tracking at TSO layer under `mode="bme"`;
  voltage security stays as local constraints — which matches Manuel's
  "voltage optimisation is left to local control" position).
- **Q2** → **REVISED 2026-07-02** (after Manuel's locality clarification;
  see the §0.2 revision note): the BME Φ-gradient uses **Convention A**
  (port-frozen local own-gradient; self-marginal through the price term;
  J = all zones incl. self). Only H_{b,i} remains supra-local — served as
  full-network **values** behind the access-enforcing
  `RestrictedSensitivityProvider`, as the simulation stand-in for what each
  TSO would identify locally (own moves + boundary measurements).
  `mode="bme"` therefore does **not** restrict `local_sensitivities_tso`;
  the zone's own loop may keep its Ward Jacobian (004 trade-off, orthogonal
  to BME correctness). Concession and realism story: §0.5 and §0.11.
- **Q3** → ✅ `g_q_tie` forced to 0 under `mode="bme"` (fail-fast if explicitly
  set non-zero).
- **Q5** → ✅ delegated: ε-acceptance applies to MIQP integers; MSC/MSR
  integrator banks keep their own commit rule but emit switch notices and
  ledger entries.

### 0.11 Architectural placement (Manuel's question, 2026-07-02)

Manuel asked whether the shared loss objective means "the shared combined model
NEEDS to be known", contradicting the standing assumption that every TSO only
knows its own area's model/Jacobian, and whether BME is therefore a **tertiary
layer** above the (local-information) secondary layer.

Recorded answer — *functionally yes, structurally no, and the model concession
is narrower than "the combined model"*:

1. **No combined model at runtime.** Φ = Σ Φ_i is never evaluated by any runtime
   entity; each Φ_i uses only zone i's own measurements (own branch currents for
   the form-B loss term, own bus voltages for the band) and own H rows. Only the
   test oracle and the Phase 6 rung (d) hold a global model, by design.
2. **The single supra-local object is H_{b,i} = ∂v_b/∂u_i** (spec §3.9's
   concession): its *values* embed how neighbouring networks respond, but it is
   (i) only a slice — response of jointly observable boundary voltages to the
   zone's *own* actuators; no neighbour topology, parameters, states or
   objectives are exchanged; and (ii) self-identifiable: zone i could estimate
   exactly this object from its own input moves plus boundary measurements
   (Kalman/RLS online-estimation line — the realism story). "Knowing a slice of
   the true plant response" ≠ "knowing the neighbour's model".
3. **Hierarchy placement:** BME occupies the classical **tertiary voltage
   control role** (system-wide, economically motivated loss minimisation above
   the zone-local secondary function) — a useful chapter framing. But
   structurally it is **not** a supervisory controller above the zones (the
   v_ref coordinator was one): it is peer-to-peer, embedded as one gradient
   term inside the existing Layer-1 MIQPs plus a message bus. What *does* sit
   "above" is contractual, not a controller: the one-time agreement on Φ's
   weights, the ownership convention (D1) and the boundary registry.
4. Cadence note: classical tertiary control is slower than secondary; BME v1
   runs at the Layer-1 cadence (with delay d and discrete slotting as the only
   temporal structure). A slower BME cadence is a possible Phase 6 variant, not
   a default.

### 0.8 Contribution claims to carry into the chapter (recorded per spec)

- **Integer externality pricing** (§3.6): H_{b,i} carries OLTC/shunt columns, so
  the local MIQP prices the committed integer move itself — vs relax–round–resolve.
- **Phulpin distinction** (§3.3): single common Φ, no inter-TSO exchange rate.
- **v_ref → BME narrative** (§3.7): v_ref (setpoint consensus, validated) is the
  intuition; BME the formal generalisation; the two documented v_ref-era flaws
  (incommensurability, sticky-OLTC) are precisely what Φ and discrete hygiene fix.

### 0.9 Non-goals (future work, per spec §1)

- Vertical propagation of DSO-level switch notices to neighbouring TSOs.
- Angle-based boundary coupling.
- Online estimation of H_{b,i} (tie-in: `sensitivity_updater` SMW machinery and
  `numerical_h` exist; Kalman/RLS with directional forgetting is the realism
  story for the concession row of §0.5).

### 0.10 Go / no-go

**GO — unconditional as of 2026-07-02.** All decisions D1–D8 and open questions
Q1/Q2/Q3/Q5 resolved with Manuel (§0.7); audit outcomes accepted (Convention
**B**, J = neighbours only; `mode="vref"` = existing tie coordinator kept as a
separate path; B = tie-endpoint buses with overlapping B_ij at IEEE bus 3).

The three highest implementation risks, in order: (i) gradient/weight rescaling
when Φ replaces g_v-scale tracking (D2 — the v1 price failure showed exactly
this failure mode); (ii) the distributed-equals-centralised gradient test
tolerance under the shared-Jacobian + measurement pipeline; (iii) interaction
of slotting with existing cooldowns (D5).

---

## Phase 1 — Boundary topology and sensitivities ✅ (2026-07-02)

New modules (each with a §3-symbol header map, British English, fail-fast):

| File | Content |
|---|---|
| `network/boundary_topology.py` | `BoundaryTopology`, `TieLine` — registry B (fixed ascending order), B_ij, tie orientation zone_i < zone_j, own/adjacent boundary per zone, **closure-based ownership** (D1: every in-service bus owned by exactly one zone; generator-terminal and orphan buses inherit the zone they are electrically embedded in), tie-loss shares 50/50, **separator assertion** (hard error; cross-zone non-line branches also raise with the "enlarge B" message of §3.2) |
| `sensitivity/boundary_sensitivity.py` | `ZoneInputSpec` (u_i column structure `[Q_DER \| Q_PCC_set \| V_gen \| s_OLTC \| s_shunt]`, bus-level DER columns — per-DER expansion stays controller-side), `RestrictedSensitivityProvider` (assembles H_{b,i} from the shared full-network Jacobian, mirroring the controller's column conventions incl. the PCC load-convention negation and the shunt V² step scaling), `ZoneBoundaryView` (zone-bound handle exposing ONLY `h_b()`; out-of-scope access raises `PermissionError` — §3.9 made enforceable) |
| `sensitivity/marginal_computer.py` | `MarginalComputer` — area-internal reduced Jacobian with the zone's own boundary buses as ports: R = −J_int⁻¹ · ∂g_int/∂V_port (θ_port fixed, D7 magnitudes-only); `response_v()` = ∂v_int/∂v_b, `response_full()` (θ and V, for the Phase 2 loss gradient), `mu(grad_v_int, grad_direct)` embedding into registry order with **exactly-zero** entries outside the zone's adjacent set (§3.4 sparsity, enforced); pinned ports handled via the ∂g/∂V voltage-source column; 3W star buses of zone-owned trafos included in the interior block |

Tests — `tests/test_boundary_topology.py`, `tests/test_boundary_sensitivity.py`,
`tests/test_marginal_computer.py`: **26 passed** (plus the existing
`test_sensitivity_updater.py` re-run green after the package-export additions
to `network/__init__.py` and `sensitivity/__init__.py`).

Acceptance criteria (spec §5 Phase 1):

- ✅ Separator check passes on the 3-area IEEE 39 case; a deliberately broken
  partition raises (boundary bus dropped from its zone → spanning component),
  and a synthetic cross-zone trafo raises.
- ✅ FD validation of H_{b,i} columns on the 3-area case: V_gen (all three
  zones, ≤5 % + 1e-5), Q_DER (≤5 %), one **whole OLTC tap step** and one whole
  shunt step (secant-vs-tangent, ≤15 %). Q_PCC_set columns share the
  Q-injection primitive (negated load convention) and are covered by the same
  code path; a runner-net FD with real 3W PCC couplers lands with Phase 4's
  integration tests.
- ✅ FD validation of μ per area: synthetic quadratic Φ over interior PQ buses,
  port sub-network oracle (each own-boundary bus pinned by a voltage source at
  the plant operating point), central differences per port — agreement ≤2 %.
  Bonus: the §3.2 separator *consequence* is asserted numerically first (the
  port sub-network reproduces every interior plant voltage to <1e-6 pu).
- ✅ Sparsity: μ entries at non-adjacent boundary buses are exactly zero;
  direct terms outside the adjacent set raise.
- ✅ `RestrictedSensitivityProvider` raises (`PermissionError`) on out-of-scope
  access; the zone view exposes no other read surface; returned matrices are
  copies (cache cannot be poisoned).

Findings / corrections made during Phase 1:

1. **Slack correction** (propagated to §0.4 and D1): the system slack is a
   ``slack=True`` gen at a 10.5 kV terminal bus (pp bus 40, the ppc reference)
   behind boundary bus 38's machine trafo — bus 38 itself is PQ, its H/μ rows
   are live. `pinned_boundary_buses` is empty on this case; the pinned-bus
   handling remains implemented for nets where a boundary bus genuinely hosts
   a voltage source.
2. **Bus 19 (IEEE 20) mystery resolved**: `build_ieee39_net` REMOVES it — the
   two-trafo chain 19–20–34 is collapsed into one machine trafo
   (`network/ieee39/build.py` l. 249–291). Documented by test.
3. **Slack-machine actuator exclusion**: the slack gen and its machine trafo
   are not zone actuators (no Jacobian column at the reference bus) — matching
   the runner's `ZoneDefinition` convention; the provider fail-fasts if they
   are requested.
4. The adjacent-boundary sets overlap as expected (IEEE bus 3 in B_12 ∩ B_23);
   μ registry embedding handles this without special-casing.

Carried to later phases: real-PCC-coupler FD (Phase 4 integration nets);
OOS-gen/OLTC column masking hooks (Phase 4, wired from controller state);
loss-gradient chaining through `response_full()` (Phase 2 — ✅ done).

---

## Phase 2 — Common objective ✅ (2026-07-02)

New / extended modules (§3-symbol header maps, British English, fail-fast):

| File | Content |
|---|---|
| `controller/common_objective.py` (NEW) | `CommonObjective`: Φ_i = w_loss·P_i^loss + Σ φ_band (§3.3; D1 ownership — interior branches by endpoint owner, tie lines split 50/50 via `tie_loss_shares()`, N_i^own = `topology.zone_buses(i)`), the φ_band C¹ hinge (value + gradient), `phi_global()` computed INDEPENDENTLY of the ownership map (the invariant oracle), and `gradients(comp)` → `ZoneGradients`: μ_i assembly (§3.4) and the Convention-A g_own primitives `d_q_injection` / `d_pcc_set` / `d_vgen` / `d_tap_2w` / `d_shunt` (§3.5). w_band has NO default (D2 leaves the magnitude to Phase 6; 0.0 = losses-only ablation rung) |
| `sensitivity/marginal_computer.py` (extended) | `mu_x()` — μ from a gradient over the FULL internal state (θ and V, aux 3W star states included), chained through the existing R; `mu()` now delegates to it. `frozen_input_response()` + `response_to_q_injection` / `response_to_vgen` / `response_to_tap_2w` — the port-frozen operators ∂x_int/∂u|_{v_b fixed} = −J_int⁻¹·∂g_int/∂u (J_int kept from the Phase 1 build; ownership-enforced, §3.9 locality; hv-side taps only, fail-fast otherwise) |
| `sensitivity/index_helper.py` (extended) | `get_ppc_line_index()` (mirrors `get_ppc_trafo_index`, positional within `net.line`) |
| `controller/__init__.py` | exports `CommonObjective`, `PhiBreakdown`, `ZoneGradients` |

Formulation notes:

- **Values** use the ACTUAL branch losses from the res tables
  (`res_line/res_trafo/res_trafo3w.pl_mw`) — the controller's form-B
  monitored-line surrogate (Q4) is NOT used for Φ values; Q4 is thereby
  resolved for the invariant (full owned-branch coverage by construction).
- **State-space loss gradient** dP^loss/d(θ, V) is analytic from the ppc
  branch model (weighted row-sums of the MATPOWER dSbr_dV identities over
  `Yf`/`Yt`; module header carries the formulas), converted to MW on the
  system base. Reconciliation ppc-losses == res-tables verified to machine
  precision on the 3-area case before implementation.
- **μ_i** = R-chained interior gradient (`mu_x`) + direct terms ∂Φ_i/∂v_b
  at adjacent boundary buses: owned-branch/tie-share loss sensitivity at
  own ports AND far tie endpoints; band gradient at own ports only (D1 —
  the far endpoint's band belongs to its owner).
- **g_own (Convention A)**: ∇_{x_int}Φ_iᵀ·(port-frozen response) plus the
  explicit input terms — V_gen: pinned-terminal loss + band sensitivity
  (the terminal magnitude IS the input; no μ channel, but a live g_own
  channel — the Phase-1 "PV rows inert" note applies to μ only); OLTC:
  closed-form ∂P^loss_ℓ/∂τ of the transformer's own branch; Q/PCC/shunt:
  none (state-only). PCC columns are the negated injection (load
  convention); shunt columns carry the −q_step·V² scaling — both mirror
  `RestrictedSensitivityProvider`/controller conventions exactly.

Tests — `tests/test_common_objective.py`: **24 passed** (full BME suite 60;
controller-side regression `test_tso_output_gradient` / tie-coordination /
loss-objective suites re-run green — 32 passed).

Acceptance criteria (spec §5 Phase 2):

- ✅ **Partition invariant** Σ_i Φ_i == Φ_global at the base point + 20
  randomised operating points (loads ±10 %, gen setpoints ±0.01 pu, seed
  123), ≤1e-9 relative, for BOTH rungs (losses-only w_band=0 and an active
  tight band w_band=100 @ [0.99, 1.01]). Independent oracles: Φ_global
  (w_band=0) equals the raw res-table totals to 1e-12; zone 1's tie split
  recomputed by hand (lines 2, 14, 25 at 50 %).
- ✅ **FD of ∂Φ_i/∂v_b**: μ_i vs central differences at EVERY adjacent
  boundary bus (Phase 1 port-subnet oracle extended by the far tie
  endpoints, all adjacent buses pinned) — ≤2 %, both objective rungs; own
  ports exercise the internal-response chain, far endpoints isolate the
  direct tie-share terms. Sparsity: exact zeros outside the adjacent set.
- ✅ **Hinge behaviour**: zero inside and AT the edges, quadratic outside,
  gradient continuous (C¹), one-sided curvature 2·w_band / 0 at both edges.
- ✅ (beyond the letter of §5, per §0.2 revision (ii)) **Convention-A g_own
  primitives FD-validated** with ports pinned: Q injection ≤5 % (2 interior
  buses/zone), V_gen ≤5 % (1 gen/zone), OLTC whole-step secant ≤15 %
  (incl. trafo 0 whose hv bus IS a boundary port — port rows correctly
  frozen), synthetic 20 Mvar shunt step ≤15 %. Locality (`PermissionError`
  analogue: foreign actuators raise), PV-bus injection raises, slack
  machine excluded, PCC negation pinned.

Findings / corrections made during Phase 2:

1. `JacobianSensitivities._compute_dg_dtau_2w` ALREADY existed (l. 2696,
   tie-coordination era) returning `(dg_dtau, Δτ, dQ_direct)` — reused as-is
   (a briefly added duplicate was removed; `sensitivity/jacobian.py` is
   net-unchanged by this phase). `dQ_direct` (τ-dependence of Q observed at
   a PV endpoint) is not needed for Φ: the explicit τ-dependence of the
   losses enters through the branch-loss term instead.
2. **Q7 (NEW, open — Manuel's call, interacts with D2):** N_i^own =
   `zone_buses()` closure includes the 10.5 kV machine-terminal buses (their
   band terms are live via V_gen inputs), and on RUNNER nets it would also
   include the DN feeder buses under the PCC couplers — i.e. φ_band would
   penalise DN voltages at TSO level, and Φ would include DN losses. The
   spec's literal N_i^own reading is implemented (include-all); whether the
   band set should be restricted (e.g. by voltage level) is a Phase 4/6
   config decision to surface BEFORE the first runner integration.
3. Tap direct term uses the exact ppc ratio (τ from `branch[:, 8]`, hv/lv
   side handled); the indirect ∂g/∂τ part keeps the repo's τ = 1 + s·Δτ
   convention — identical on this net (ratio exactly 1.0, all taps hv-side,
   verified) and policed by the whole-step FD tests either way.

Carried to later phases: real-PCC-coupler / 3W-loss FD on runner nets
(Phase 4 integration tests — 3W branch weights and aux-star loss gradients
are implemented but the base net has no trafo3w); Q7 decision; OOS-element
masking hooks (Phase 4).

---

## Phase 3 — Coordination bus and signals ✅ (2026-07-02)

New module `core/coordination_bus.py` (beside the vertical message classes
of `core/message.py`; §3-symbol header map, British English, fail-fast):

| Component | Content |
|---|---|
| `MarginalSignal` / `SwitchNotice` | Spec §4 frozen dataclasses, repo-adapted (`zone_id: int`, `step`): registry-order μ + v_b^meas snapshot / dv_b^pred + device tuple. Vectors validated (1-D, finite, registry length) and frozen read-only on construction |
| `CoordinationBus` | In-process pub/sub for ≥2 zones: integer delay d ≥ 0 (D4; message published at k is visible at k+d, NOT earlier — and signals are per-step, no stale carry-over), optional drop probability ∈ [0, 1] with a bus-owned seeded RNG (> 0 without a seed raises); drop decisions are drawn AT PUBLISH TIME, one per (message, receiver) in ascending receiver order → pattern depends only on the publish sequence, never on query order/repetition. Self-delivery never happens. Duplicate (zone, step) marginal publishes raise; multiple notices per step are allowed (one per committed move). Structured `drop_log` |
| `MarginalReceiver` | Receiver-side §3.4 low-pass μ^filt = (1−β)·μ^filt + β·μ(k−d) per SENDER (D3 β = 0.3; β = 1 disables smoothing — identity-test configuration); must be stepped consecutively (gaps raise — a skipped step would corrupt the filter cadence). Returns `ReceivedMarginals(step, coordinated, mu_neighbour_sum)`; **the self-marginal never touches bus or filter** (Convention A — the controller adds μ_i locally, Phase 4) |
| Explicit §3.8 policies | Cold start: exactly d steps `coordinated=False`, one `cold_start` event each. Warm + missing signal: RAISES when drops disabled (protocol violation); with drops enabled → hold-last-FILTERED-value (`hold_last` event per occurrence). First signal dropped → no state to hold: contributes exactly zero until first arrival (`extended_cold` event per occurrence — documented policy, not a silent default). Filter initialisation: first received sample (β = 1 once), documented |

Tests — `tests/test_coordination_bus.py`: **15 passed** (pure in-process,
no pandapower; full regression sweep incl. Phases 1–2 and core-importing
suites: 97 passed).

Acceptance criteria (spec §5 Phase 3):

- ✅ Delay semantics: published at k → visible at k+d for every other
  zone, empty before, per-step (nothing at k+d+1 unless published at
  k+1); d = 0 same-step exchange works (needed by the Phase 4 identity
  test); notices share the delay and never return to the sender.
- ✅ Cold start: logs and runs uncoordinated for exactly d steps, then
  the first warm step sums the first samples.
- ✅ Missing-signal-after-warm-up raises when drops are disabled.
- ✅ Hold-last-value: under p = 0.5 (seed 7) every warm step's neighbour
  sum is reproduced from the event log + filter states — held senders'
  states are bit-identical frozen, delivered senders follow the exact
  β-recursion, never-arrived senders contribute zero; p = 1 gives a
  zero neighbour sum with `extended_cold` logged per sender and step.
- ✅ Determinism: same seed → identical drop log and identical filtered
  sums; different seed → different pattern.

Design notes recorded:

- Expected senders default to ALL other zones (not only tie-adjacent
  ones): the price term H_{b,i}ᵀ·Σ_j μ_j spans all of B, so every zone's
  marginal is relevant — sparsity lives in the μ vectors, not in the
  routing.
- A dropped NOTICE is lost (logged), not held — it is an event, not a
  state; hold-last only applies to the μ state channel.
- The bus is runner-agnostic (§3.9): zones will interact with the bus
  and the plant only; Phase 4 wires `publish → receiver.update` into the
  per-step sequence and maps the spec §4 `coordination:` block onto
  `MultiTSOConfig` fields.

---

## Phase 4 — Controller integration 🚧 (2026-07-02: core done, wiring pending)

### Q7 resolved (Manuel, 2026-07-02): Φ scope = transmission level

Φ (band AND losses) covers only the transmission system level, not the
subordinate distribution networks. Implemented as an explicit
`CommonObjective(vn_kv_min=...)` scope: buses with vn_kv ≥ threshold
carry the band; a branch contributes losses only if EVERY terminal bus
is in scope. For IEEE 39 with vn_kv_min = 220: the 345 kV lines plus the
two 345/345 kV interconnectors; machine trafos (345/10.5), generator
terminal buses and (on runner nets) everything below the PCC couplers
are excluded. Ownership (D1) unaffected — scope selects terms, not
owners. Gradient side is weight-driven (the excluded machine trafo's tap
remains an actuator whose only Φ effect is the indirect 345 kV response;
its explicit ∂P_ℓ/∂τ term carries weight 0). Default 0.0 = spec-literal
include-all (kept for the generic invariant tests); the BME experiment
configs set the TS threshold. Tests: TS-scope partition invariant, μ FD,
V_gen/tap FD, manual value semantics (`test_common_objective.py`, 29
green).

### D7 REVISED (2026-07-02, forced by the hard-gate test): complex boundary

**Finding.** The first run of §3.5 test 2 FAILED (zone-1 DER at bus 24:
g_bme −2.87e-3 vs FD −3.46e-3 MW/Mvar, 17 %). Diagnostic by elimination
(scratch, recorded numbers): the single-area TOTAL analytic gradient
matches FD to 0.001 % — every Phase 2 gradient piece is exact — while
the magnitude-only identity misses by exactly the boundary-ANGLE
channel (dθ_b/dQ ≈ 1e-5 rad/Mvar at nearby boundary buses × loss angle
sensitivity O(10–100 MW/rad)). A loss objective is strongly
angle-coupled; the D7 magnitudes-only design cannot satisfy the exact
identity `dΦ/du_i = ∂Φ_i/∂u_i|_{v_b} + H_{b,i}ᵀ·Σ μ_j`.

**Resolution — D7's pre-authorised fallback applies** ("complex voltages
… admissible architectural fallback if the design ever requires it",
Manuel, §0.7): boundary coordinates are now STACKED
``[Vm_b (registry) | θ_b (registry)]`` ∈ R^{2|B|} everywhere in the
exchange path:

| Component | Extension |
|---|---|
| `MarginalComputer` | θ-port columns ∂g_int/∂θ_port beside the V columns (a port at the reference bus raises); `mu_x_stacked()` → μ ∈ R^{2|B|}. The V-only `mu()`/`mu_x()` remain (Phase 1/2 semantics + oracles unchanged) |
| `RestrictedSensitivityProvider` | `h_b_stacked(zone)` (2|B| × n_i) assembled from full state responses per column class; magnitude rows cross-checked ≡ the legacy helper-based `h_b` to 1e-9 (test); angle rows non-trivial. `ZoneBoundaryView.h_b_stacked()` is the second permitted read — the §3.9 informational scope is UNCHANGED (still the zone's own columns at jointly observable boundary buses, now both coordinates) |
| `ZoneGradients` | `mu_stacked()`; θ direct terms = w_loss·(loss angle-gradient) at adjacent buses (φ_band has no θ channel) |
| `CoordinationBus` usage | `n_boundary = 2|B|`; `MarginalSignal.v_b_meas` carries the same stacking (no signal-class change) |

Realism note for the chapter: the θ_b rows/entries presume boundary
angle observability (PMU at boundary substations) — strictly MORE
defensible than internal observability, but stronger than the pure
V-measurement story; tie-in with the existing PMU stub of the loss
objective work. Record as an explicit assumption.

### New: `controller/bme_gradient.py`

`BMEGradientAssembler` (per zone): `g_own()` over the ZoneInputSpec
columns `[Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]` from the
Phase 2 primitives (bus-level DER; per-DER expansion stays
controller-side), `mu()` = stacked μ_i, `g_bme(mu_total)` = g_own +
h_b_stackedᵀ·μ_total with μ_total = μ_i (local, undelayed, unfiltered)
+ receiver neighbour sum. `pcc_hv_buses()` helper mirrors the
provider/controller PCC resolution. Single-area degenerate mode:
`MarginalComputer` now allows a portless zone IFF the topology has no
boundary (nothing frozen → total-response operators, μ empty); a
portless zone in a multi-zone topology still raises.

### Hard-gate identity tests ✅ — `tests/test_bme_gradient_identity.py` (15 green)

- **§3.5 test 2 (the money test)**: stacked distributed gradients ==
  FD of global Φ w.r.t. stacked u, with the neighbour μ routed through
  the REAL CoordinationBus/MarginalReceiver (d = 0, β = 1) and the self
  term added locally. Full column coverage (DER, PCC stand-in
  345/345 kV trafo, V_gen, OLTCs incl. the port-hv trafo 0, 20 Mvar
  shunt) at the base point + 2 randomised OPs; plus 10 further
  randomised OPs (reduced columns). Continuous ≤5 %, whole-step
  discrete ≤15 % (secant vs tangent), objective = Q7 TS scope with
  active tight band.
- **§3.5 test 1 (single-area identity)**: one-zone partition → no
  boundary, μ = ∅, price = 0, the "port-frozen" own gradient IS dΦ/du —
  verified against FD (DER/V_gen/tap).
- Assembler validation (zone mismatch, wrong μ length).

### Wiring progress (2026-07-03)

1. ✅ **Config** — `MultiTSOConfig` gains the spec §4 block as flat
   fields: `coordination_mode` ("none"|"vref"|"bme", default "none"),
   `bme_delay_steps=1`, `bme_drop_probability=0.0`, `bme_beta_filter=0.3`,
   `bme_seed=None`, `bme_w_band=0.0`, `bme_v_soft_min_pu/max_pu`
   (±3 %), `bme_vn_kv_min=220.0` (Q7). Documented mutual exclusions
   ("bme" × `enable_tie_coordination`; "bme" × non-zero g_q_tie)
   fail-fast in controller/runner, not silently reconciled.
2. ✅ **TSOController hook** (injection pattern, minimal diff, byte-
   identical when unused): `enable_bme_mode()` (raises on non-zero
   `g_q_tie`, Q3), `receive_bme_gradient(g_bus_level)` (one-shot,
   validated), `_bme_objective_gradient()` (per-DER expansion
   ∇_der = [Eᵀ·∇_bus(DER); rest] via the existing DER mapping — WITHOUT
   touching the H expansion cache), and a single branch at the top of
   `_compute_objective_gradient`: under `bme_mode` the private objective
   gradient is fully REPLACED by the injected g_i^bme (D2/Q1); output
   constraints, CAIR, integer handling untouched. Missing injection
   under bme mode raises (per-step sequence enforcement). Controller
   regressions re-run green (32).

3. ✅ **Runner wiring** (2026-07-03, `multi_tso_dso.py`):
   - Setup (after the tie-coordinator block): mode validation —
     "vref" requires `enable_tie_coordination=True` (alias for the
     existing gated path, unchanged); "bme" fail-fasts on
     `enable_tie_coordination`, on `numerical_h` (needs the analytic
     shared Jacobian) and — **v1 scoping decisions, revisit later** — on
     `local_sensitivities_*=True` (the runner freezes reduced Jacobians;
     the Ward-loop variant stays future wiring) and on
     `refresh_shared_jac_on_tso=False` (μ_j is defined at the measured
     state of step k, §3.4 — v1 realises this by re-linearising
     shared_jac each TSO tick; measurement-evaluated gradients on a
     frozen model are noted future work). Builds BoundaryTopology (from
     `tn_zone_map`), CommonObjective (config weights/edges/Q7 scope),
     per-zone ZoneInputSpec from `ZoneDefinition` (bus-level DER columns
     from the controller's DERMapping first-seen unique order; raises on
     duplicate DER buses without a mapping), `enable_bme_mode()` per
     zone, CoordinationBus (stacked 2|B| coordinates) + MarginalReceiver
     per zone.
   - Per TSO tick (inside the `run_tso` branch, after measurements,
     before the zones solve): rebuild provider + MarginalComputer +
     ZoneGradients + assembler at the freshly re-linearised shared_jac;
     compute μ_i (stacked), publish (v_b_meas = stacked [vm | θ_rad]
     snapshot), `receiver.update(tso_step_count−1)`; μ_total = μ_i +
     neighbour sum (cold start: self term only, logged by the
     receiver); `receive_bme_gradient(g_bme)` into each zone controller.
   - Suites re-run green (56: output-gradient invariant, tie
     coordinator + hooks, identity hard gate, bus).

### End-to-end validation runs ✅ (2026-07-03 — Phase 4 closed)

Scenario: 005 CIGRE cascade config shortened to 30 sim-minutes (90 plant
steps, 10 TSO ticks), headless.

- **mode="none" == pre-BME baseline, BITWISE.** Same run from a
  worktree of commit `0d7c47b` (last pre-BME-code commit) vs current
  HEAD: 90-step loss trajectories identical to the last bit
  (max |Δ| = 0.0). The solver stack is deterministic, so the spec's
  strong reading holds.
- **vref regression, BITWISE.** `enable_tie_coordination=True` +
  `coordination_mode="vref"` (the alias) reproduces the pre-BME tie
  coordination run exactly (max |Δ| = 0.0).
- **mode="bme" end-to-end smoke: runs.** Full per-step chain executes
  on the real cascade net (3W PCC couplers, DN feeders — first live
  exercise of the Q7 scope and the 3W machinery): 90 steps, 36.1 s vs
  32.5 s baseline (~10 % per-tick machinery overhead), cold start
  logged, no failures. Behavioural calibration is EXPECTED to be open:
  the MW-scale Φ gradient against G_w weights tuned for the g_v = 1e7
  private objective makes near-zero moves — the `gw_precondition`
  rescaling (risk #1) is the Phase 6 rung-configuration task, not a
  wiring defect.
- Smoke config requirements found: the 005 config runs
  `local_sensitivities_* = True` — the bme rung must set both False
  plus `refresh_shared_jac_on_tso=True` (v1 validation raises
  otherwise, as designed).

*(Phase 5 section follows after the Phase 4 correction note below.)*

**Phase 1 finding CORRECTED (via the smoke run): the slack machine IS a
zone actuator.** The runner's `ZoneDefinition` carries the slack
machine's AVR setpoint (gen at the reference bus) and its machine trafo
(12) as ordinary actuators — the Phase 1 "slack not an actuator"
exclusion was wrong for the runner convention. BME support added:
`response_to_vgen` accepts the reference bus (the slack magnitude is an
exogenous PF input with well-defined ∂g/∂V_ref — the missing STATE
column never mattered for this input channel), and a tolerant ∂g/∂τ
assembly (`dg_dtau_2w_tolerant`, mirrors `compute_dV_ds_2w`'s
accumulate-existing-rows behaviour) replaces the raising helper for tap
responses and the stacked H_b rows. The identity hard-gate test now
covers BOTH new column types (zone 1 full spec: slack V_gen + trafo 12)
— FD-confirmed. The Phase 2 `test_slack_machine_not_an_actuator` was
revised accordingly (`test_slack_machine_vgen_supported`).

---

## Phase 5 — Discrete hygiene ✅ (2026-07-03)

New module `controller/discrete_hygiene.py` (§3-symbol header, fail-fast)
plus a minimal solve-path hook:

| Component | Content |
|---|---|
| `SlottingSchedule` | §3.8.2 / D5 round robin (ascending zone ids, slot length in TSO steps, default 1): `slot_owner(tick)` / `may_commit(zone, tick)` — exactly one committer per tick, deterministic |
| `epsilon_accepts()` | §3.8.3 / D6 rule: accept the discrete part iff Φ̂(MIQP) − Φ̂(QP_frozen) ≤ −(ε_switch + c_switchᵀ·\|Δu_d\|), with Φ̂ = the per-step MIQP/QP objective values (the local quadratic model). ε = 0 default = pure improvement sign test (the frozen QP optimum can never beat the MIQP optimum, so ε = 0 ≈ always-accept while still ledgering) |
| `SwitchingLedger` / `LedgerEntry` | Append-only §3.8.3 ledger: step, zone, device labels (`oltc:`/`shunt:` + u-index), Δu_d, predicted ΔΦ̂, one-time deferred realised ΔΦ, accepted flag, reason (`accepted` \| `epsilon_reject` \| `slot_blocked` \| `integrator_commit`), slot owner, ε, cost. `to_records()`/`from_records()` round-trip (Phase 6 parquet surface) |
| `BaseOFOController` hook | Step 7b: `result = self._post_solve_gate(result, solve_frozen=...)` — default no-op (byte-identical; mode="none" BITWISE regression RE-VERIFIED after this change). `_solve_with_frozen_integers()` re-solves the identical per-step problem with every integer pinned at its current value (raises if infeasible — the incumbent point must be feasible) |
| `TSOController` gate | Overrides `_post_solve_gate` under armed hygiene: no integer move → pass-through (frozen solve not even called); slot context (one-shot, runner-fed via `set_bme_slot`; missing context raises) → `slot_blocked` returns the frozen result; else ε-acceptance → accept (MIQP result) or `epsilon_reject` (frozen result); every decision appended to the shared ledger; `bme_ledger_indices_this_step` exposes the rows for the deferred realised-ΔΦ fill. Armed via `configure_bme_hygiene()` (requires BME mode) |
| Config | `bme_slotting` (True), `bme_slot_length` (1), `bme_epsilon_switch` (0.0), `bme_switch_cost_oltc/shunt` (0.0) — magnitudes are the D6 Phase 6 calibration |
| Runner wiring | Setup: shared ledger + schedule, `configure_bme_hygiene` per zone. Per tick: (i) deferred ledger fill — realised ΔΦ = Φ_global(now) − Φ_global(previous round) for last tick's entries (simulation-oracle privilege, §3.10.2 premise data); (ii) notice consumption at k (delay d, drops apply) feeding `bme_notice_mask_hook` — the §3.8.1(b) estimator-masking hook, a DOCUMENTED no-op in v1 (no online estimator; shared_jac re-linearised each tick, so there is no innovation to correct); (iii) slot context injection per zone; (iv) after the zone solves: `SwitchNotice(dv_b_pred = H_{b,i}^d·Δu_d` in stacked coordinates`)` published for committed discrete moves. BME internals (`bme_ledger`, `bme_bus`, …) exposed through the `pre_loop_hook` state dict |

Tests — `tests/test_discrete_hygiene.py`: **14 passed** (full regression
sweep 139; mode="none" bitwise regression re-verified; bme smoke with
hygiene armed reproduces the exact pre-hygiene trajectory — no integer
moves at the uncalibrated Φ scale, gate inert as expected).

Spec §5 Phase 5 test mapping:

- ✅ ε-acceptance rejects a constructed marginal-benefit switch and
  accepts a constructed clear-benefit switch (+ slot-blocked overrides a
  clear benefit; no-move short-circuit; missing-context raise).
- ✅ Ledger schema round-trips; append-only semantics enforced.
- ✅/🔶 Two-area counter-switch scenario: implemented at the GATE level
  (both zones propose discrete moves on the same tick: without slotting
  both commit, with slotting exactly the slot owner — deterministic).
  The CLOSED-LOOP plant variant (a scenario engineered so the plant
  actually counter-switches) belongs to the Phase 6 scenario work.
- 🔶 "Notice correction changes the neighbour's innovation as
  predicted": structurally DEFERRED — v1 has no innovation to correct
  (no online estimator; per-tick re-linearisation). The hook exists and
  receives every delivered notice (spec's own "no-op if no online
  estimator active" clause); transport semantics are pinned by the
  Phase 3 bus tests. Revisit with the online-estimation tie-in.

Carve-out (fail-fast, not silent): `coordination_mode="bme"` with
`shunt_dispatch="integrator"` RAISES — Q5 requires integrator-bank
commits to emit notices + ledger entries (`integrator_commit` reason is
reserved), and that emission is not wired into the integrator commit
path yet (sign-sensitive, currently untestable — the bme rung runs the
MIQP shunt path). To be wired when a bme rung first enables integrator
banks.

---

## Phase 6 — Evaluation ladder + Monte Carlo 🚧 (started 2026-07-03)

Work items (handover list): (1) w_Φ calibration ✅ §6a; (2) D6 ε/c_switch
calibration; (3) w_band + soft-edge sweep incl. w_band = 0 ablation;
(4) oracle rung (d); (5) metrics module completion (gap-to-oracle,
Phulpin fairness, oscillation indicator); (6) MC campaign. Ladder script:
`experiments/011_BME_LADDER.py` (rungs none / vref / bme / bme_loss;
shared 005 scenario; uniform Φ metric via `record_bme_phi`).

### 6a — w_Φ calibration (`bme_gradient_scale`) ✅ (2026-07-03)

**Mechanism decision (supersedes the "gw_precondition rescaling" phrasing
of the earlier plan):** risk #1 is closed by a single scalar
`MultiTSOConfig.bme_gradient_scale` (w_Φ) applied to the ENTIRE injected
BME gradient (g_own and price term alike). This is algebraically a units
choice Φ′ = w_Φ·Φ — exchange-rate-free (one common objective, D2/Phulpin
distinction intact), it leaves G_w identical across ladder rungs (clean
comparison; no per-actuator trust-region reshaping), and the identity
tests are invariant. Consequence recorded in the config docstring:
ledgered ΔΦ̂ predictions are in the SCALED units — the D6 ε/c calibration
(item 2) must be performed in those units.

**Robustness fixes required to run the calibration** (the 60-min CIGRE
scenario trips gen 2 at minute 60; the bme rung — `refresh_shared_jac_on_
tso=True` — is the first configuration whose Jacobian ever SEES a machine
outage; three layers fixed, daily log
`2026-07-03_bme_phase6a_calibration_oos_fixes.md`):
1. loss-gradient ppc/ppci branch alignment via `branch_is` masking
   (`controller/common_objective.py`, previous session);
2. `actuator_active()` OOS masking — disconnected actuators keep their
   u-column but contribute exactly-zero H_{b,i} columns / g_own entries
   (`sensitivity/boundary_sensitivity.py`, `controller/bme_gradient.py`,
   previous session, verified here);
3. **pre-existing latent bug** (this session): the Q_gen row block of
   `TSOController._build_sensitivity_matrix` passes the UNFILTERED
   generator list to the `compute_dQgen_*matrix` primitives, which
   indexed a pruned terminal bus straight into the internal arrays.
   Fixed in `sensitivity/jacobian.py`: `_ppc_bus_is_internal()` guard in
   all four matrix functions (+ NaN guard in the shunt variant) —
   disconnected machine ⇒ physically exact zero row/column. Bitwise-safe
   for `mode="none"`/`vref`: their frozen time-0 Jacobian never contains
   pruned buses (the pre-fix `none` reference completed the identical
   scenario); regression suite re-run green.

**Calibration sweep** (60-min CIGRE incl. the gen-2 trip; losses-only Φ,
w_band = 0, d = 1, slotting on; metric = sustained total losses
(mean last 10 steps) + run V extremes as stability proxy):

| w_Φ | losses first/last/mean₁₀ [MW] | V range [pu] | verdict |
|---|---|---|---|
| none (ref) | 29.58 / 52.61 / 32.92 | [0.978, 1.049] | reference |
| 1e4 | 29.65 / 54.19 / 33.49 | [0.968, 1.044] | inert-to-noise (worse than ref) |
| **1e5** | 29.62 / 50.03 / **31.00** | [0.986, 1.059] | **chosen**: −5.8 % sustained, V contained |
| 1e6 | 29.29 / 46.93 / 30.16 | [0.991, 1.140] | −8.4 % but V escapes the band |
| 1e7 | 28.71 / 46.75 / 30.36 | [1.002, 1.179] | over-driven; 3.7× runtime (solver stress) |

**Outcome: w_Φ = 1e5** (filled into `011_BME_LADDER.py::
BME_GRADIENT_SCALE`) — the largest swept scale whose voltage envelope
stays contained *without* the band hinge; delegated calibration call,
open to Manuel's veto.

**Two findings for the record:**
- The 1e6/1e7 voltage escape is the empirical confirmation of D2's
  design argument: a losses-only common objective drives voltages up,
  and the repo default `g_z_voltage = 1e-12` is the known inert
  placeholder that relied on g_v tracking (same trap as the 2026-07-01
  heterogeneous-strategies corridor bug). The `bme_loss` (w_band = 0)
  ablation rung must therefore be read with this caveat — or given a
  binding `zone_g_z_voltage` (config decision for the D2 sweep, not
  made silently here).
- The w_Φ = 1e7 rung located the over-drive edge (the v1-price failure
  mode) at ~100× the chosen scale — useful margin knowledge for the MC
  sensitivity sweep.

**Validation of the real bme rung** (w_band = 1e3, w_Φ = 1e5, 60 min incl.
the gen-2 trip; `011_BME_LADDER.py --rung bme --minutes 60`):
- V ∈ [0.982, 1.045] — the band hinge pulls the envelope back INSIDE the
  uncoordinated reference's own range ([0.978, 1.049]); compare the
  losses-only rung's [0.986, 1.059] at the same w_Φ.
- Sustained losses 31.92 MW (mean last 10) vs none 32.92 (−3.0 %); the
  losses-only ablation reaches −5.8 % — the gap is the (measured) price
  of the voltage-security margin. Φ mean(last 10) = 22.82 MW.
- **Discrete hygiene is exercised at this scale for the first time**:
  16 ledger entries — 6 accepted, 10 slot-blocked, 0 ε-rejects (ε = 0);
  5 OLTC switches; runtime 129 s / 180 steps (no solver stress).
- Metric caveat: `band_violation_frac = 1.0` against the ladder's
  0.97–1.03 proxy band — the CIGRE schedules themselves sit at ~1.03+,
  so the reference rungs will show the same; interpret this metric only
  ACROSS rungs once the full ladder runs.

### 6b — First full ladder (360 min) + D6 calibration ✅ (2026-07-03)

**Ladder results** (`011_BME_LADDER.py --rung all --minutes 360`; the full
005 case-study horizon incl. gen-2 trip @60′/restore @180′, 200 MW +
100 Mvar load step @90′–360′, tie-line-25 trip @260′/restore @360′;
last-hour means; hygiene at ε = 0, i.e. pre-D6):

| rung | Φ [MW] | losses [MW] | Δlosses vs none | OLTC switches | V range [pu] | ledger acc/ε-rej/slot |
|---|---|---|---|---|---|---|
| none | 37.47 | 47.10 | — | 5 | [0.978, 1.055] | — |
| vref | 37.48 | 46.99 | −0.2 % | 4 | [0.981, 1.054] | — |
| bme | 36.42 | 45.64 | **−3.1 %** | 26 | [0.982, 1.063] | 24/0/33 |
| bme_loss | 30.98 | 40.80 | (−13.4 %) | 33 | [0.986, **1.170**] | 30/0/57 |

Readings:
- **vref is neutral** on this uniform-schedule scenario (−0.2 %) — exactly
  its documented behaviour (nothing to decouple); whatever the bme rungs
  deliver is attributable to the common-objective mechanism.
- **bme (with band): −3.1 % sustained losses with the voltage envelope
  essentially held** (1.063 vs the baseline's own 1.055 — slight excess,
  input to the D2 w_band sweep).
- **bme_loss's −13.4 % is INADMISSIBLE**: V reaches 1.170 pu over the long
  horizon (the 60-min calibration's 1.059 was deceptive — rising load +
  6 h of accumulation). This is the strongest empirical vindication of
  the D2 band-hinge design so far, and the ablation rung's story for the
  chapter: losses-only common objectives buy loss reduction with voltage
  security. (The default `g_z_voltage = 1e-12` backstop is inert — the
  open D2-sweep decision on `zone_g_z_voltage` stands.)
- Switch counts 26–33 vs baseline 5 at ε = 0 — the D6 calibration below
  exists precisely to close this gap (spec headline claim 4).

**D6 calibration** (from the bme rung's 57-entry ledger, scaled Φ̂ units):
- anchor (b) = median per-step |ΔΦ| on no-commit steps = 1039 scaled
  (0.0104 MW);
- **ε_switch = 5×(b) = 5193 ≈ 5.2e3** — the independent sanity cap
  0.5 × median |ΔΦ̂_proposal| = 5256 lands on the same value (two
  derivations agree);
- **c_oltc = 1.0e3 (1×(b)); c_shunt = 5.2e3 (5×(b))** — breaker vs tap
  wear, mirroring the shunt integrator's stricter dwell/budget treatment
  of bulk devices.
- Premise data (§3.10.2): predicted-vs-realised sign agreement 0.78 over
  23 filled accepted switches on a window containing four contingency
  events (1.00 on the clean 60-min run); realised magnitude ≈ 60 % of
  predicted on the clean window. Recorded honestly — the MC campaign
  extends this distribution.
- Constants wired into `011_BME_LADDER.py` (both bme rungs — the
  ablation isolates w_band only). Ledger ΔΦ̂ units are w_Φ-scaled, as
  required.

**Post-ε re-run** (identical scenario, hygiene at the calibrated
ε = 5.2e3, c_oltc = 1.0e3, c_shunt = 5.2e3; last-hour means):

| rung | Φ [MW] | losses [MW] | Δ vs none | OLTC switches | ledger acc/ε-rej/slot | V range [pu] |
|---|---|---|---|---|---|---|
| bme | 36.48 | 46.13 | −2.1 % | **19** (was 26) | 17/**15**/52 | [0.982, 1.063] |
| bme_loss | 31.45 | 41.62 | (−11.6 %) | 23 (was 33) | 19/**36**/106 | [0.986, 1.156] |

Readings:
- The ε-gate does its designed job: 15/36 marginal proposals rejected,
  switching −27 % (bme) / −30 % (bme_loss) for ≈1–2 pp of the loss gain.
- Spec headline claim 4 ("switch counts at or below the baseline") is
  NOT yet met at this ε (19 vs the baseline's 5) — deliberately not
  over-tuned on a single scenario; the MC ε-sweep is the instrument for
  mapping the ε ↔ (switching, Φ) trade-off, and this pair of runs is its
  first two points.
- bme_loss remains voltage-inadmissible post-ε (1.156 pu) — ε governs
  switching, not the voltage escape; only w_band does. Consistent.

### 6c — D2 (w_band × edges) sweep ✅ (2026-07-03)

Nine-point sweep at the **120-min horizon** (Manuel's directive: short
calibration horizons — covers the gen-2 trip @60′ and the load step
@90′; last-hour metric = the post-contingency hour; reference `none`
derived from the 360-min records' first 120 min: 60.65 MW, V ∈ [0.978,
1.049]). Edge families: spec default (0.97, 1.03) — hinge ACTIVE at the
~1.03 operating schedule, contrary to the "zero in normal operation"
design intent; wide corridor (0.95, 1.05); and Manuel's
operating-point-centred proposals (1.01, 1.05) / (1.00, 1.06).

| edges, w_band | losses [MW] | Δ | V range [pu] | switches | runtime |
|---|---|---|---|---|---|
| (0.97, 1.03), 1e2 | 57.57 | −5.1 % | [0.994, 1.065] | 7 | 139 s |
| (0.97, 1.03), 1e3 | 58.27 | −3.9 % | [0.992, 1.056] | 7 | 136 s |
| (0.97, 1.03), 1e4 | 59.30 | −2.2 % | [0.977, 1.048] | 11 | 1294 s |
| (0.95, 1.05), 1e2 | 57.23 | −5.6 % | [0.994, 1.086] | 10 | 886 s |
| (0.95, 1.05), 1e3 | 57.37 | −5.4 % | [0.994, 1.070] | 10 | 809 s |
| (0.95, 1.05), 1e4 | 57.58 | −5.1 % | [0.994, 1.065] | 13 | 864 s |
| **(1.01, 1.05), 1e3** | 57.09 | −5.9 % | [0.995, 1.072] | 11 | 451 s |
| **(1.01, 1.05), 1e4** | **57.38** | **−5.4 %** | [**1.002**, 1.062] | 10 | **410 s** |
| (1.00, 1.06), 1e3 | 57.19 | −5.7 % | [0.994, 1.078] | 10 | 359 s |

**Outcome (Manuel, 2026-07-03): edges (1.01, 1.05), w_band = 1e4** —
his operating-point-centred corridor dominates: 2.6× the loss gain of
the 6b pairing at nearly the same V_max, the best post-trip voltage
support of all nine points (lower hinge lifts the dip to 1.002 pu vs
the baseline's 0.978 — the band genuinely acts as disturbance support),
and the healthiest bme solve times. Honest caveat recorded: no centred
pairing strictly contains V_max at the baseline's 1.049 — only the
tight (0.97, 1.03)×1e4 does, at −2.2 % and 9× runtime.

Findings for the chapter:
- **Soft hinges do not cap** — they slow excursions (1.05-edge families
  reach 1.062–1.086 depending on w_band); strict containment needs
  either a stiff tight band (costly) or the MIQP hard constraints.
- **Solve time is a diagnostic**: pairings whose hinge fights the loss
  gradient at the operating point (tight edges at high w_band) or whose
  band leaves voltages roaming (weak corridor) run 6–9× slower than the
  centred corridor.
- **Uniform-Φ-metric fix**: `make_ladder_config` now sets the D2 band
  definition (w_band + edges) on EVERY rung so the recorded Φ is the
  identical functional across none/vref/bme (previously the non-bme
  rungs recorded losses-only Φ — their 6b Φ column understated bme's
  advantage if anything). The bme_loss CONTROL ablation necessarily
  zeroes w_band for its gradient, so its recorded Φ differs by
  definition — compare it on the losses column.
- Ablation-rung backstop question (from 6b) resolved by rationale: the
  bme_loss rung keeps NO voltage backstop — the escape IS the ablation's
  finding; a `zone_g_z_voltage` would change what the rung demonstrates.

Also in 6c: **per-zone Φ_i recording** (`bme_phi_zone_mw` in the
records, filled via `CommonObjective.phi_zone`) — the premise data for
the Phulpin normalised-overcost fairness metric (item 5); partition
invariant verified live on the runner net (Σ_i Φ_i = Φ_global to 1e-6).

### 6d — Oracle rung (d): single-zone BME oracle ✅ (2026-07-05)

**D8 interpretation (Manuel, 2026-07-03: "both"):** rung (d) is the
**single-zone BME oracle** now; the V5-style full-set Φ oracle (DSO DERs
+ 3W coupler taps in the solve, no DSO cascade) stays the optional
additional bound for after the MC campaign.

**Design.** `MultiTSOConfig.single_zone_partition = True` collapses the
partition to ONE zone = the union of the fixed 3-area TN bus sets (the
identical bus/actuator universe as the distributed rungs), combined with
`coordination_mode="bme"`. Then there are no ties, the boundary registry
is empty, the port-frozen operators degenerate to total-response
operators and **g_bme = dΦ/du exactly** — the single-area identity (spec
§3.5 test 1, pinned by `test_bme_gradient_identity.py`) *is* the
oracle's correctness proof. One MIQP per step over all TSO-layer inputs
(incl. every PCC setpoint), global Φ, no communication, DSO cascade
unchanged below, same solver/step logic and the same D6 hygiene
(slotting degenerates to a single always-committing owner). Rung wired
as `011_BME_LADDER.py --rung oracle`.

**Changes.** Config field `single_zone_partition`; runner: partition
branch, HV-network/tertiary-shunt/DSO-map zone remaps (`_hv_zone`),
degenerate BME path (no bus/receivers when |zones| = 1 — the
CoordinationBus's two-zone fail-fast stands for real multi-zone runs);
`RestrictedSensitivityProvider` now accepts an EMPTY registry (zero-row
H_{b,i}; the all-pinned fail-fast for non-empty registries stands).
Regression: 37 tests green (boundary sensitivity, gradient identity,
discrete hygiene); the ladder figures pick the oracle up automatically.

**Scenario-identity bug caught by the first 120-min run (2026-07-05):**
the zonal generator P dispatch received the CONTROL partition, so the
single-zone flag silently changed the *plant scenario* (one system-wide
residual-load balance instead of three per-zone ones: Φ_first 64 vs ~21
MW on every other rung; losses nearly doubled). Fixed: the dispatch now
always uses the fixed 3-area partition (`dispatch_zone_map`), the
single-zone flag affects the control layer only (spec §6 shared-scenario
rule). Verification: post-fix Φ_first = 19.65 MW — in family. The bad
run's artefacts were overwritten by the re-run.

**Result (120-min horizon, D2-final pairing, D6 hygiene):**

| rung | losses last-hr [MW] | Δ vs none | V range [pu] | switches | ledger acc/ε-rej/slot |
|---|---|---|---|---|---|
| none | 60.65 | — | [0.978, 1.049] | — | — |
| bme (distributed) | 57.379 | −5.4 % | [1.002, 1.062] | 10 | — |
| **oracle** | **57.380** | −5.4 % | [1.005, 1.065] | 23 | 10/0/0 |

**Headline finding (spec claim 2): the distributed BME closes ≈100 % of
the gap to the centralised oracle on this scenario** — sustained losses
identical to three decimals, near-identical voltage envelopes — and with
FEWER discrete switches (the oracle, with no slotting and no staleness,
commits 23 tap moves vs the distributed rung's 10; its ledger shows the
single-decision-maker signature 10 accepted / 0 slot-blocked). The runs
differ genuinely (trajectories, ledgers, runtimes — 681 s vs 410 s);
they converge to the same continuous optimum. Caveats, recorded
honestly: one scenario, one seed; the oracle keeps the per-zone G_w
blocks (same step logic per D8 — it is the *decomposition* bound, not a
retuned central controller nor an OPF bound); the MC campaign (item 6)
tests whether the ≈100 % closure survives delay/drop/H-error sweeps.

Remaining Phase 6 items: (5) metrics completion (gap-to-oracle wiring in
the summary — data now exists; Phulpin fairness from `bme_phi_zone_mw`;
oscillation indicator); (6) MC campaign (seeds, parquet; ε-sweep per 6b;
ledger = §3.10.2 premise data). Final ladder table at the D2-final
pairing = ladder `--rung all` once (5) lands; V5-Φ oracle = optional
extra bound thereafter.
