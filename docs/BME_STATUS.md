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
| 2 | Common objective | ❌ not started |
| 3 | Coordination bus and signals | ❌ not started |
| 4 | Controller integration | ❌ not started |
| 5 | Discrete hygiene | ❌ not started |
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
loss-gradient chaining through `response_full()` (Phase 2).
