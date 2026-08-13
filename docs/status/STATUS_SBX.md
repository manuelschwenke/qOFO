# SBX Minimal — Status

Scheduled Boundary Exchange (SBX Minimal), per the Claude Code Build Plan v2
(2026-07-07). Horizontal analogue of the vertical CAIR + setpoint-scheduling
mechanism: capability messages, a deterministic scheduling rule, voltage
setpoints, settlement at fixed contract prices. No price discovery. Runs side
by side with BME; BME code is not modified.

---

## 2026-07-07 — Phase 0: Reconnaissance (GATE — awaiting confirmation)

### 0.1 Located infrastructure

| Item | Finding |
|---|---|
| `rep1()` helper | **Does not exist as code.** It appears only as a convention reference ("`rep1()`-style assertions", `docs/BME_SPEC.md` §8). The codebase raises `ValueError`/`RuntimeError` with precise messages everywhere. **Proposal:** define `sbx/fail.py::rep1(msg, **diagnostics) -> NoReturn` (raises `RuntimeError` with formatted diagnostics) and use it SBX-wide. |
| Plant protocol | No formal `Protocol` class. The plant interface is the trio `pp.runpp(net)` + `core.measurement.measure_zone_tso(net, zone_def, it) -> Measurement` + `experiments.helpers.plant_io.apply_zone_tso_controls(...)`; `core/pf_adapter.py` mirrors it for PowerFactory. Controllers see only `Measurement`. SBX complies by consuming `Measurement` fields only. |
| Per-area measurement extraction, `res_line` signs | `measure_zone_tso` (`core/measurement.py:429-458`) reads tie-line Q at the **in-zone endpoint** in **load convention**: positive = Q flowing from the endpoint bus into the line (leaving the zone), via `res_line.q_from_mvar` / `q_to_mvar`. `ZoneDefinition.tie_line_indices` / `tie_line_endpoint_buses` are populated unconditionally by the runner (`multi_tso_dso.py:776-777`, from `network.zone_partition.get_zone_tie_lines`). **Gap:** tie-line **P is not measured** — SBX Step 1 needs per-line cycle-averaged P (`p_sched_mw[ℓ]`). See gate question G2. |
| OFO tracked-output mechanism | `TSOControllerConfig.v_setpoints_pu` (one reference per monitored voltage bus) with **scalar** weight `g_v`; gradient `2·g_v·(V − V_set)ᵀ·∂V/∂u`. Runtime-update precedents: `TSOController.receive_tie_coordination` (`tso_controller.py:860`) and `update_voltage_setpoints` (`tso_controller.py:1152`). Corridor terminal buses are monitored buses of their zone (same precondition the vref path asserts). Consequence: per-bus tracking weights do not exist and adding them would touch the MIQP assembly (forbidden) → **`w_track` ≡ the zone's `g_v`** (consistent with the plan's config note "align with existing tracked-output weights"). |
| Vertical CAIR dataclass style | The vertical path itself (`core/message.py::CapabilityMessage`) is a plain class, but the BME bus (`core/coordination_bus.py::MarginalSignal`) established the target style: `@dataclass(frozen=True)`, validated/immutable numpy fields, fail-fast `__post_init__`. SBX messages follow the frozen-dataclass style (plan requires "frozen, versioned"). |
| LP access in the solver wrapper | `optimisation.miqp_solver.MIQPSolver._solve_qp` is exactly the capability-LP shape with `G_w = 0` (zero quadratic → pure LP, `OSQP`/`ECOS` backends), `G_z = 0` (selects the **hard** output-constraint branch, `miqp_solver.py:409-452`), `alpha = 1`, `grad_f = ∓ H_term^T s_corr`, box bounds on `u`, output bounds on `v`. No solver-wrapper change needed. **Caveat:** `_solve_qp` returns a non-optimal status instead of raising (`MIQPResult.status`, zeros in `w`) — `sbx/capability.py` must check `result.is_optimal` and `rep1` otherwise. |
| Scenario generator & MC harness | `experiments/006_CIGRE_MONTECARLO.py::run_one_scenario` (SimBench profile-year start time + lightly-constrained random contingency schedule, drop-and-replace acceptance, schedules persisted to CSV); reused by `experiments/012_BME_MONTECARLO.py` in a paired-scenario design (identical scenario across arms). SBX Phase 7 mirrors 012 with arms `{AUTONOMOUS, SBX, BME}`. |
| Step-time semantics → `k_sched` | BME experiments (`011`/`012`) use the 005 config: `dt_s = 20 s`, `dso_period_s = 20 s`, `tso_period_s = 180 s`. One TSO OFO iteration = **3 min** → **`k_sched` = 5 TSO iterations per 15-min cycle** (5 measurement samples per cycle average). Per-cycle quantum at defaults: `dq_quant = 10 Mvar × (15 min / 15 min) = 10 Mvar`. |
| Runner integration point | `experiments/runners/multi_tso_dso.py`: `coordination_mode ∈ {"none", "vref", "bme"}` validated at `~1103-1156`; horizontal coordination fires in the TSO-step block (`~3300-3390`, vref pattern: coordinator update → messages → `receive_tie_coordination`). SBX adds `coordination_mode = "sbx"` **in the runner only** (the runner is not on the do-not-touch list; BME modules, vertical CAIR path, MIQP assembly, solver wrapper remain untouched). |

### 0.2 BME IEEE 39 partition and corridor table

Fixed 3-area partition (`network.zone_partition._FIXED_ZONES_IEEE39`, 0-indexed
TN buses), `BoundaryTopology` builds cleanly: separator assertion passes, **no
cross-zone trafo/trafo3w/impedance** (every inter-area branch is a line).
Boundary registry B = {1, 2, 8, 13, 14, 16, 17, 26, 38}.

Base case for the table below: `build_ieee39_net(scenario="wind_replace",
ext_grid_vm_pu=1.03)`, bare TN network (no HV sub-networks, no profiles),
converged `pp.runpp`. S_base = 100 MVA, all terminals 345 kV. Per-unit line
parameters (100 MVA base; `parallel = 1` throughout; `b_sh` = total line
charging, `g_sh = 0` for all ties):

| Corridor (A,B) | Line idx | Terminal A (zone) | Terminal B (zone) | r [pu] | x [pu] | b_sh [pu] | Base V_A / V_B [pu] | Base P_A / Q_A (at A end) |
|---|---|---|---|---|---|---|---|---|
| (1,2) | 2 | bus 1 (z1) | bus 2 (z2) | 0.00130 | 0.01510 | 0.25720 | 1.0220 / 1.0152 | +373.4 MW / +10.9 Mvar |
| (1,2) | 14 | bus 38 (z1) | bus 8 (z2) | 0.00100 | 0.02500 | 1.20000 | 1.0287 / 1.0455 | +71.2 MW / −134.7 Mvar |
| (1,3) | 25 | bus 26 (z1) | bus 16 (z3) | 0.00130 | 0.01730 | 0.32160 | 1.0166 / 1.0160 | −10.4 MW / −11.7 Mvar |
| (2,3) | 5 | bus 2 (z2) | bus 17 (z3) | 0.00110 | 0.01330 | 0.21380 | 1.0152 / 1.0143 | −32.8 MW / −1.5 Mvar |
| (2,3) | 18 | bus 13 (z2) | bus 14 (z3) | 0.00180 | 0.02170 | 0.36600 | 1.0128 / 1.0038 | +26.9 MW / +21.0 Mvar |

Corridor flows at the reference end (export from the smaller-id area
positive): `q_corr(1,2) = −123.8 Mvar`, `q_corr(1,3) = −11.7 Mvar`,
`q_corr(2,3) = +19.6 Mvar`.

Note: the experiment base case (HV sub-networks attached, profile start time,
zonal dispatch) shifts these values; contract defaults (`v_std`, `q_std`
consistency) are constructed at runtime from the converged experiment base
case, not from this bare-TN snapshot. Line 14 (bus 38–8) carries heavy
charging (b_sh = 1.2 pu) and imports ~135 Mvar into zone 1 at the base point —
the corridor (1,2) standard flow is dominated by line charging, not by
voltage-difference transfer.

### 0.3 Topology check (corridor level)

- The 3-area corridor graph is a **triangle**: edges (1,2), (1,3), (2,3).
  Per plan §4 Phase 0: **the 3-area configuration is excluded from v2 scope**
  (bilateral externalities would be an observed metric only if opted in
  later). Not a string, so no 3-area SBX experiments in v2.
- **No 2-area partition exists in the repo.** The 2-area headline
  configuration must be constructed by merging two of the three fixed zones
  (the zone machinery is partition-agnostic — the oracle rung already runs a
  1-zone merge). Options:
  - **(a) A = zone 1, B = zones 2∪3 (recommended):** one corridor, 3 tie
    lines (2, 14, 25); terminals A = {1, 26, 38}, B = {2, 8, 16}. Area A
    holds the slack (bus 38) and the fewest buses — the natural
    "self-contained north vs south" split with the richest corridor.
  - (b) A = zones 1∪2, B = zone 3: one corridor, 3 tie lines (5, 18, 25).
  - (c) A = zones 1∪3, B = zone 2: one corridor, 4 tie lines (2, 5, 14, 18).
- **Decision point 4 note (terminal impedances):** corridor-side terminals
  are *not* electrically adjacent. E.g. corridor (1,2) zone-1 side {1, 38}
  connect via bus 0 (two line hops); zone-2 side {2, 8} are several hops
  apart. Under merge (a) the acting-side terminal set {1, 26, 38} (or
  {2, 8, 16}) spans the area's whole border. The common-`dv` shift is kept
  per the plan default; Phase 5 must watch for per-terminal tracking-error
  asymmetry and report if the common shift proves inadequate.

### 0.4 ASSUMPTIONS

- **A1 (fail-fast helper).** `rep1()` will be *created* in `sbx/fail.py`
  (does not pre-exist); all SBX modules use it. → Gate G1.
- **A2 (tie-line P measurement).** `Measurement` gains an *optional,
  additive* field `tie_line_p_mw` (default empty, populated by
  `measure_zone_tso` next to the existing tie-line Q block, same endpoint
  and convention). `core/measurement.py` is not on the plan's do-not-touch
  list and the change is backwards-compatible, but it is an interface
  extension → Gate G2 (hard rule 5: "if an interface must change, stop and
  report").
- **A3 (cycle length).** `k_sched = 5` TSO iterations (15 min at the 3-min
  TSO period of the shared 005 scenario). Cycle averages use the 5 TSO-step
  measurement samples of the elapsed cycle. → Gate G4.
- **A4 (tracking weight).** `w_track` ≡ the zone's existing `g_v` (scalar
  per controller; per-bus weights would require touching the MIQP
  assembly). SBX writes corridor-terminal references into
  `v_setpoints_pu` via the same pathway as `receive_tie_coordination`.
- **A5 (capability LP).** Reuses `MIQPSolver._solve_qp` with `G_w = 0`,
  `G_z = 0`, `alpha = 1` — the hard-constraint LP branch. SBX raises on any
  non-optimal status.
- **A6 (2-area configuration).** Merge option (a) (zone 1 vs zones 2∪3)
  unless Manuel prefers (b)/(c). → Gate G3.
- **A7 (contract defaults).** `v_std` per tie terminal = converged
  *experiment* base-case voltages rounded to 1e−3 pu; golden test 4
  (`q_std` reproduces the base corridor flow) is evaluated against that same
  converged state.
- **A8 (frequency base).** Line-charging conversion `c_nf_per_km → b_sh`
  uses `net.f_hz` (verified: **60 Hz** for the IEEE 39 build — a first-pass
  50 Hz assumption during recon was wrong by ×1.2 and has been corrected in
  the table above). `sbx/tie_line_model.py` must read `net.f_hz`, never
  hard-code the frequency.
- **A9 (message transport).** SBX peer messages are exchanged in-process
  through the scheduler (both areas simulated in one runner loop), mirroring
  how vref/BME exchange works; the checksum protocol still runs on canonical
  serialisations to prove determinism.
- **A10 (settlement output location).** Ledger + Markdown settlement summary
  go under the experiment's `results/<experiment>/` directory, following the
  011/012 output convention.

### 0.5 Gate questions (answer before Phase 1 implementation proceeds)

- **G1:** Approve creating `sbx/fail.py::rep1()` (helper does not pre-exist).
- **G2:** Approve the additive `Measurement.tie_line_p_mw` +
  `measure_zone_tso` extension (interface change, backwards-compatible).
- **G3:** Choose the 2-area merge — recommendation: **A = zone 1,
  B = zones 2∪3** (option a).
- **G4:** Confirm `k_sched = 5` TSO iterations ≙ 15 min (3-min TSO period).
- **G5:** Confirm `w_track ≡ g_v` (no per-bus weight without MIQP-assembly
  changes).

Phase 0 complete. **Paused at the Phase 0 gate.**

---

## 2026-07-07 — Phase 0 gate resolution (Manuel) + PLAN AMENDMENT v2.2

Gate outcomes:

- **G2 approved:** additive optional `Measurement.tie_line_p_mw`, populated by
  `measure_zone_tso` next to the tie-line Q block (same endpoint, load
  convention). Implemented when Phase 2 needs `p_sched` (not before).
- **G3 superseded by the v2.2 amendment below:** **no zone merging.** The
  3-area triangle partition, byte-identical to BME, is the headline
  configuration. §0.3's merge options are obsolete.
- **G1/G4/G5:** no objection raised; assumptions A1 (`sbx/fail.py::rep1`),
  A3 (`k_sched = 5`), A4 (`w_track ≡ g_v`) stand.

### PLAN AMENDMENT — SBX Minimal v2.2 (recorded verbatim; replaces v2.1
item 2 "round-robin corridor slotting"; v2.1 items 1, 4, 6 stand)

Decision: all corridors execute protocol Steps 1–5 in PARALLEL every cycle.
No slotting, no intra-cycle ordering, no extra message rounds. Joint
feasibility is guaranteed at the source, in `capability.py`.

1. **D13 (replaced) — joint-box capability.** Per area i with corridor set
   C_i, solve ONE LP per cycle:

   ```
   max t >= 0
   s.t. for every sign vector sigma in {-1,+1}^{|C_i|}, exists du_sigma:
        u_min <= u + du_sigma <= u_max
        v_min + margin <= v_meas + H_loc @ du_sigma <= v_max - margin
        s_corr_c . (H_term_c @ du_sigma) = t * sigma_c * dq_quant_c
                                                      for all c in C_i
   ```

   Offer per corridor c: `offer_range_mvar = (-a_c, +a_c)` with
   `a_c = min(t, 1) * dq_quant_c`. Rationale: the box with these half-widths
   is inscribed in the joint feasibility polytope (convexity: vertex
   feasibility implies box feasibility), so ANY combination of accepted
   deals across the area's corridors is jointly achievable. |C_i| = 2 on
   the triangle → 4 vertices, one small LP (4 du blocks), existing LP
   solver wrapper. If the measured point violates the margined voltage
   limits, skip the LP and offer (0, 0) on all corridors (consistent with
   the need flag being set).
2. Protocol, messages, matching: UNCHANGED from v2. Matching remains a
   pure function of the two corridor messages; checksum protocol intact.
   An area may be acting on multiple corridors simultaneously; the
   acting-side invariant is per corridor and unaffected.
3. `voltage_margin_pu` doubles as the absorber for within-cycle terminal-
   voltage shifts induced by the area's own granted requests (partner-side
   acting) and by neighbour deals on other corridors. Phase 5 must verify:
   margin ≥ worst observed quantum-induced terminal shift in the smoke
   test; report the observed ratio.
4. Phase 3 acceptance (amended): joint-box LP returns t ≥ 1 in the
   unstressed base case (full quantum offers on all corridors); t shrinks
   monotonically under increasing stress (3-point check); t-LP never fails
   silently; offers on all corridors of a violating area are (0, 0).
5. Phase 5 smoke test (amended): per-corridor deal cadence = every cycle
   (no slotting latency criterion). Add: construct one scenario where two
   corridors of the same area accept deals in the SAME cycle; assert both
   resulting schedules are tracked without violating that area's
   constraints (this is the joint-box guarantee under test).
6. Retained from v2.1: no zone merging; 3-area triangle partition
   byte-identical to BME is the headline; `spillover_mvar` metric per
   corridor and cycle; Phase 7 on identical partition/scenarios/seeds.
7. Config: no new fields. Note in code: static per-corridor budget split
   is the trivial degenerate variant of D13 (fixed t); do not implement
   as an option.

*(Note: the full v2.1 amendment text was not supplied to this session; its
items 1, 4 and 6 are treated as standing per Manuel's instruction, with
item 6's content restated in point 6 above. If v2.1 items 1/4 contain
further binding changes beyond what points 1–7 restate, they must be added
here.)*

---

## 2026-07-07 — Phase 1: tie-line + corridor model (GATE — awaiting
## confirmation)

### 1.1 Delivered

- `sbx/fail.py` — `rep1()` fail-fast helper (`SBXError`, formatted
  diagnostics; A1/G1).
- `sbx/config.py` — frozen `SBXConfig`, plan §5 defaults; `k_sched = 5`,
  `tso_period_s = 180 s` → derived `t_cycle_min` and per-cycle
  `dq_quant_mvar` (rate-scaled). v2.2 adds no fields.
- `sbx/tie_line_model.py` — `TieLineParams` (per unit on `net.sn_mva` /
  terminal `vn_kv`; `b_sh_pu`/`g_sh_pu` are the PER-END half values of the
  symmetric π model, so the plan's Q-equation applies as written);
  `extract_tie_line_params` (handles `length_km`, `parallel`, std-type
  columns, reads `net.f_hz` — A8); `q_flow` (δ from `brentq` on the
  small-|δ| branch, transfer-limit assert, bracket assert);
  `v_sched_for_q` (nested `brentq`, bracket assert with endpoint Q values);
  `sensitivities` (total derivatives, central FD 1e−4 pu / 1e−2 MW,
  step-halving consistency ≤ 1 % with an absolute floor of 1e−6 for
  vanishing derivatives). The P-equation carries the `g_sh` term the plan
  omits (all IEEE 39 ties have `g_sh = 0`; golden test 1 validates).
- `sbx/corridor.py` — registry from the global net against the BME area
  partition (asserts: no cross-area trafo/trafo3w/impedance, two distinct
  areas per tie, non-empty registry); `Corridor` frozen dataclass
  (reference end A = smaller area id, lines ascending by `line_idx`);
  `corridor_q_flow`, `corridor_solve_dv` (Step-4 scalar root find, bracket
  assert), `corridor_sensitivities` (per-line triplets + per-side
  common-shift sums = the capability LP's `s_corr`).
- `tests/sbx/test_golden_tie_line.py` — golden tests 1–4 + registry
  ground-truth cross-check + corridor-sum consistency.

**Clarification vs plan §2.2 Step 4 (recorded, no behavioural ambiguity
left):** the corridor root find always evaluates q_corr at the reference
end A (where `q_sched` is defined) and applies `dv` to the acting side's
terminals whichever end that is. The plan's
`q_flow(v_std[far], v_std[act] + dv, ·)` reads literally only when B acts;
evaluating at the acting end when A acts would redefine the schedule by
the line losses/charging.

### 1.2 Golden-test results (gate)

Base case: `build_ieee39_net(scenario="wind_replace")`, bare TN net, fixed
3-area partition, converged `pp.runpp`.

| Test | Result |
|---|---|
| Registry vs ground truth (3 corridors, 5 ties) | **pass** |
| 1 — base-case `q_flow` vs measured, every tie line, tol max(0.5 Mvar, 1 %) | **pass** |
| 2 — line round trip `v_sched_for_q` → `q_flow` ≤ 1e−6 Mvar (targets q, q ± 5) | **pass** |
| 2 — corridor round trip `corridor_solve_dv` → `corridor_q_flow` ≤ 1e−6 Mvar (both acting ends, targets q ± 10) | **pass** |
| 3 — sensitivities vs perturbed re-run (loads × 1.01), ≤ 5 % rel (floor 0.1 Mvar) | **pass** |
| 3b — corridor sums ≡ per-line sums | **pass** |
| 4 — contract-default consistency at **1e−3 pu** rounding | **FAIL** (see below) |

### 1.3 Gate finding: §2.1 contract rounding vs golden test 4

With `v_std` = base-case voltages rounded to **1e−3 pu** (plan §2.1), the
evaluated standard deviates from the measured base-case corridor flow by
far more than the golden tolerance:

| Corridor | q_meas [Mvar] | q_std @1e−3 [Mvar] | deviation [Mvar] | tol [Mvar] |
|---|---|---|---|---|
| (1,2) | −123.821 | −119.763 | **+4.058** | 1.238 |
| (1,3) | −11.739 | −9.961 | **+1.778** | 0.500 |
| (2,3) | +19.560 | +20.473 | **+0.913** | 0.500 |

This is structural, not a bug: the tie series reactances are ~0.013–0.025
pu, so dQ_A/d(V_A − V_B) ≈ |b| ≈ 40–75 pu ≈ **4–7.5 Mvar per 1e−3 pu** of
rounding-induced voltage-difference error. Golden tests 1–3 prove the
model is exact against pandapower; test 4 cannot "hold by construction"
at 1e−3 pu granularity. Consequence if kept: the base case itself starts
up to ~4 Mvar away from `q_std` — a large bite out of the 5-Mvar tier-1
band before any deal happens, silently biasing tier-2 surplus and tier-3
attribution.

**Options (decision needed before Phase 2 freezes the contract data):**
1. **Round `v_std` to 1e−4 pu (recommended).** Expected deviation scales
   ~×10 down (≈ 0.1–0.4 Mvar) — inside every golden tolerance; 0.1 mpu is
   still a realistic precision for an agreed operating-point datum.
2. Keep 1e−3 pu and redefine test 4's reference: compare future `q_std`
   evaluations against the *contract-evaluated* standard rather than the
   measured base flow (internally consistent — both sides evaluate the
   identical function — but the standard is then biased vs the physical
   base point, with the band-erosion consequence above).
3. No rounding (store `v_std` at full precision; contract data less
   "round" but exact by construction).

`tests/sbx/test_golden_tie_line.py::test_golden_4_contract_default_consistency`
is left encoding the plan's 1e−3 rounding and therefore **fails** until
the decision lands (1 failed, 6 passed).

Phase 1 complete apart from the test-4 decision. **Paused at the Phase 1
gate.**

### 1.4 Gate resolution (Manuel, 2026-07-07): contract rounding

Manuel: "round to 1e-4 first". Empirical outcome at 1e−4 pu: corridors
(1,2) and (2,3) pass (dev −0.525 / +0.146 Mvar) but **(1,3) fails
marginally** (dev −0.562 vs 0.500 Mvar tolerance) — worst-case rounding on
the stiff single-line corridor (|b| ≈ 58 pu → up to 0.58 Mvar per 1e−4 pu
of voltage-difference error, i.e. 1e−4 cannot *guarantee* the 0.5-Mvar
floor for these ties). Following the "first" escalation: **1e−5 pu
adopted** — worst deviation 0.067 Mvar, every corridor ≥ 85 % margin.
Normative constant: `sbx/contract.py::V_STD_DECIMALS = 5` (decision trail
in its docstring). Golden test 4 green at 1e−5.

---

## 2026-07-07 — Phase 2: contract data and standard schedule ✅

Delivered:

- `sbx/contract.py` — `CorridorContract` (frozen dataclass, scalar/tuple
  fields only; `dataclasses.FrozenInstanceError` on mutation);
  `build_default_contract` (v_std from the converged base case rounded to
  `V_STD_DECIMALS`, constants from `SBXConfig`; raises on an incomplete
  `net.res_bus`); `q_std_mvar` (Step-1 standard: pure function of the
  frozen contract and the cycle's `p_sched`, `assert_matches` guards
  contract/corridor alignment); rate-scaled per-cycle `dq_quant_mvar`.
- `core/measurement.py` — the approved G2 extension: additive optional
  `Measurement.tie_line_p_mw` (same length, ordering and load convention
  as `tie_line_q_mvar`), populated by `measure_zone_tso` from
  `res_line.p_from_mw`/`p_to_mw` at the in-zone endpoint. No signature
  reordering (appended keyword), no existing caller changes.
- `tests/sbx/test_contract.py` — acceptance: default contract reproduces
  the base-case corridor flow (golden test 4 through the contract path);
  immutability; quantum rate-scaling (15-min and 3-min cycles);
  contract/corridor mismatch raises; stale-result guard raises;
  `tie_line_p_mw` population vs `res_line`; default-empty backward
  compatibility.

Test state: `pytest tests/sbx` → **14 passed**. Regression check on the
measurement-touching suites (`test_measurement`, `test_controller`,
`test_tie_coordination_hooks`, `test_tso_loss_objective`): 71 passed,
1 pre-existing skip — the extension is invisible to existing consumers.

---

## 2026-07-07 — Phase 3: need flag and capability ✅

Delivered:

- `sbx/need.py` — `NeedTracker` (per area; violation > `v_viol_threshold_pu`
  persisting `n_need` CONSECUTIVE iterations on the SAME direction;
  direction change or iteration gap restarts the count; deeper violation
  decides when both bounds are violated, exact tie → import);
  `NeedDecision.request_sign(own_end)` maps the area-wide direction into
  corridor-flow sign space (import need: −1 at end A, +1 at end B);
  `assert_relieving_sign` (plan §2.3 sanity assert: the local-model
  sensitivity of the worst-violated bus per unit of the signed request
  must match the relieving direction; `rep1` with diagnostics otherwise).
- `sbx/capability.py` — v2.2 D13 joint-box LP: one LP per area and cycle,
  `max t` s.t. every sign vertex σ ∈ {−1,+1}^{|C_i|} has a feasible Δu_σ
  (actuator box, margined voltage box via local H, corridor equalities
  `control_row·Δu_σ = t·σ_c·dq_c`). Mapped 1:1 onto the EXISTING
  `MIQPSolver._solve_qp` (G_w = 0, G_z = 0 hard branch, α = 1, w = [t |
  Δu_σ1 … Δu_σm]); no solver modification. Offers `(−a_c, +a_c)`,
  `a_c = min(t, 1)·dq_quant_c`. Measured point outside the margined
  limits → LP skipped, all offers (0, 0), `skipped_due_to_violation`
  marker. Non-(near-)optimal status → `rep1` (never fails silently).
- `tests/sbx/test_need_capability.py` — acceptance: no flags on the
  base case; flag fires after EXACTLY `n_need` iterations with correct
  under/over signs and correct per-end request signs; recovery/gap
  resets; relieving-sign assert both ways; joint-box t ≥ 1 with full
  quantum offers on the unstressed synthetic area; t strictly decreasing
  over a 3-point stress ramp (last point < 1 quantum); violating area →
  (0, 0) offers; forced solver failure → `SBXError`; single-corridor
  degenerate case.

Note: need tests run on real IEEE 39 base-case voltages (bounds tightened
to create deterministic violations without extra power flows); the
capability tests use an analytically checkable synthetic area. The
joint-box LP on a real zone's cached H is exercised in the Phase 5
closed-loop smoke test. Test state: `pytest tests/sbx` → **25 passed**.

---

## 2026-07-07 — Phase 4: messages and matching ✅

Delivered:

- `sbx/messages.py` — `PeerCairMessage` (frozen, versioned
  `SBX_MESSAGE_VERSION = 1`, validated `__post_init__` mirroring the BME
  bus dataclass style): `offer_range_mvar` (must contain zero),
  optional `request_mvar`, `p_sched_mw` carried by the REFERENCE-END
  sender only (enforced both ways). `canonical_serialisation()`
  (sorted-key JSON, `repr` floats — bit-exact) + SHA-256 `checksum()`;
  `assert_checksums_match` aborts the cycle via `rep1` on mismatch.
- `sbx/matching.py` — `match()` as a pure function of (msg_A, msg_B,
  contract, q_sched, q_std): mutual = sign·min(|req|), unpaid, no
  requester-of-record; unilateral = clip(request, supporter offer),
  paid, dust-rejected below `dq_min_deal_mvar`; opposite signs →
  `ScarcityEvent` (kind = "scarcity"); contract cap
  `|q_sched + dq − q_std| ≤ dq_contract_max` always enforced; request
  magnitude must equal the per-cycle quantum. `DealRecord` (frozen) with
  its own canonical serialisation + checksum for the both-sides-identical
  proof.
- **Contract addition:** `dq_min_deal_mvar` moved INTO
  `CorridorContract` (from config at default construction) — the dust
  threshold is matching-relevant and must be bilateral data for
  deterministic two-sided evaluation. (Plan §2.1 lists it under config
  only; recorded here as a deliberate schema decision.)
- `tests/sbx/test_messages_matching.py` — acceptance: unilateral full
  quantum + clip-to-offer, mutual min unpaid, opposite-sign scarcity,
  dust rejection, contract cap (both directions), quantum-magnitude
  guard, checksum round trip + mismatch abort, byte-identical
  determinism, role/cycle confusion guards, p_sched carrier rules.

Test state: `pytest tests/sbx` → **38 passed**.

---

## 2026-07-07 — Phase 5 amendments (Manuel): diagnostics and consistency

Three additions to the Phase 5 scope, recorded verbatim in intent:

1. **Border-actuator diagnostic.** Detect controllable generators / DER /
   AVRs directly at a corridor terminal bus or one transformer away;
   log them and watch the acting-side invariant with that context
   (MAVR-thesis finding: border-bus PV controllers can create
   decentralised coordination artefacts). Implemented in
   `sbx/adapter.py::_border_actuator_diagnostic` (hop 0 = at the
   terminal, hop 1 = one 2W/3W winding away); printed by the runner at
   `verbose ≥ 1` and reported by the smoke test. Diagnostic only.
2. **Smoke-test reported numbers.** R1 maximum internal voltage
   violation per zone after accepted deals (supporter areas must show
   zero — criterion C8, the joint-box guarantee made visible), R2
   maximum corridor-terminal reference-tracking error |v_meas − v_ref|,
   R3 observed_terminal_shift / voltage_margin_pu (v2.2 item 3).
3. **Post-cycle contract-consistency classification.** At every cycle
   boundary the ELAPSED cycle is classified:
   sign(q_meas − q_std) vs sign(q_sched − q_std) plus an approximate
   magnitude band (0.25–4× the surplus). Classified, never aborted —
   deadband/noise deviations (≤ q_band) and sub-dust surpluses are
   labelled as such (`sbx/scheduler.py::CONSISTENCY_*`,
   `CorridorCycleRecord.consistency`). Protects against tie-line-model
   sign errors and terminal-reference mix-ups by making them visible.

---

## 2026-07-07 — Phase 5: scheduler and control integration ✅

### 5.1 Delivered

- `sbx/scheduler.py` — six-step cycle scheduler (v2.2: all corridors in
  parallel every cycle). Surplus representation `s = q_sched − q_std`
  (q_std re-evaluated per cycle from the persistence `p_sched`, so pure
  P drift never masquerades as a deal); Steps 1–6 per plan §2.2 incl.
  deal-XOR-unwind per cycle, paid-first unwind with `m_release` dwell,
  zero-crossing accumulator resets, Step-4 invariant assert every
  cycle, double-evaluation + checksum for Step 3 (A9). Extended with
  `CorridorCycleRecord.q_meas_mvar` (cycle-averaged measured corridor
  Q) and the amendment-3 consistency classification (`CONSISTENCY_*`).
- `sbx/adapter.py` — `SBXRunnerAdapter`: builds registry + contracts +
  scheduler from the runner's live objects, feeds `record_step` every
  TSO tick, composes `AreaCycleData` at boundaries from the
  controllers' CACHED models only (voltage rows of the expanded H,
  operating-point input bounds, relieving-sign scalar from
  `compute_dV_dQ_der` at the last need decision's worst bus), and
  writes the frozen references through the EXISTING
  `update_voltage_setpoints` path (`w_track ≡ g_v`, A4/G5).
  Documented conservative capability-box adjustments: integer
  actuators frozen in the joint-box LP; Q_PCC,set box anchored at
  measured interface Q + the vertically reported DSO capability
  interval; box widened to contain the current u. Joint-box LP backend
  pinned to HiGHS via `MIQPSolver(solver="HIGHS")` (the OSQP default
  hits `user_limit` on the LP's mixed Mvar/pu column scales; no
  solver-wrapper change). Border-actuator diagnostic (amendment 1) and
  per-tick `terminal_history` (meas + refs) included.
- Runner (`experiments/runners/multi_tso_dso.py`) —
  `coordination_mode="sbx"`: fail-fast validation (excludes
  `enable_tie_coordination`, single-zone partition, `numerical_h`,
  `local_sensitivities_*`), SBXConfig resolution
  (`MultiTSOConfig.sbx_config`, `sbx_warmup_s`), adapter construction
  at the first TSO tick ≥ `sbx_warmup_s`, one `on_tso_step` call per
  TSO tick before the zones solve, exposure via the `pre_loop_hook`
  state (`sbx_runtime`). `git diff` confirms: no BME module, vertical
  CAIR path, MIQP assembly or solver wrapper touched.
- `tests/sbx/test_scheduler.py` — 7 protocol-level tests on the real
  IEEE 39 registry/contracts with synthetic capability data (steady
  state, same-cycle two-corridor deals, acting-side + invariant,
  scarcity, paid-first unwind with dwell and return to v_std, contract
  cap, off-boundary guard). `pytest tests/sbx` → **45 passed**.
- `tests/sbx/smoke_sbx_closed_loop.py` — three-arm closed-loop smoke
  test (below).

### 5.2 A7 REVISED: contracts freeze at the settled state (warmup)

The first closed-loop iteration exposed that the pre-loop "converged
base case" is NOT the closed-loop operating point: the OFO drives the
terminals toward the zones' 1.03 pu schedules within minutes, leaving
standing `q_meas − q_std` offsets of 20–70 Mvar on the stiff ties
(mpu-scale reference mismatch × |b| ≈ 40–75 Mvar/mpu) and biasing the
whole SBX arm low at the boundaries. **A7 (revised):** contract
defaults freeze at the first TSO tick at/after
`MultiTSOConfig.sbx_warmup_s` (default 30 min), i.e. at the SETTLED
closed-loop state. Outcome: `q_std` matches the measured corridor flow
within ~1–3 Mvar at freeze, and the consistency classification shows
**zero `sign_mismatch`** over all corridors and cycles.

### 5.3 Closed-loop smoke test (three arms, 360 min, PASSED C1–C8)

Scenario: 005 config; contracts frozen at min 30; 500 Mvar inductive
sink at bus 15 (zone 3) from min 60 to min 210 (10 cycles); zone-3
`v_min` tightened to 1.00 (persistent violation, calibrated); arms
`sbx` / `sbx_inert` (contract pinning, need threshold unreachable —
isolates the pinning price) / `none`, identical otherwise.

| Criterion | Result |
|---|---|
| C1 unilateral paid deals, requester = zone 3 | **PASS** (4 deals on (1,3) and (2,3)) |
| C2 same-cycle deals on both zone-3 corridors (v2.2 item 5) | **PASS** (cycles 3, 4) |
| C3 deal benefit vs sbx_inert | **PASS** (1.980 < 1.999 pu·step; see 5.4-F1) |
| C4 settling (no opposite-sign deals under stress) | **PASS** |
| C5 unwind ≤ ⌈s/quantum⌉+m_release cycles, refs at v_std | **PASS** |
| C6 every TSO solve optimal | **PASS** (all three arms) |
| C7 margin ≥ worst within-cycle terminal shift (v2.2 item 3) | **PASS** (6.37 mpu vs 10 mpu; shift/margin = 0.64) |
| C8 supporter areas violation-free after deals (joint box) | **PASS** (z1 = z2 = 0.0 exactly) |

Reported numbers: R1 worst post-deal violation depth z1/z2/z3 =
0/0/0.0153 pu; R2 worst terminal |v_meas − v_ref| = 36 mpu (the
stress-onset transient at zone-3 terminal 14); R3 shift/margin = 0.64.
Full protocol arc observed: deals c3–c4 → need clears (import + own
recovery) → dwell → unwind to zero DURING the stress → no re-request →
refs at v_std at run end. Results pickle:
`results/sbx_phase5_smoke/smoke_result.pkl`.

### 5.4 Findings for the gate (mechanism-level, need discussion)

- **F1 — pinning cost ≫ deal benefit in this scenario.** Zone-3
  exposure: autonomous 1.645, sbx_inert 1.999, sbx 1.980 pu·step.
  The deal benefit (+0.019) is real but small — the need threshold
  (5 mpu) caps relief at "violation just below flag depth" — while the
  contract pinning itself costs +0.354: holding the requester's own
  terminals at v_std (Step-4 invariant) denies zone 3 the boundary
  lift its autonomous 1.03-tracking would provide. The plan's original
  "violations strictly below the autonomous baseline" criterion is not
  achievable under this contract design on this scenario; the smoke
  gates on the deal benefit and reports the pinning cost. Candidate
  levers if F1 matters for Phase 7: preventive/deeper need trigger
  (plan §7 item 1), v_std as a negotiated schedule rather than a
  snapshot, or requester-side reference freedom within the band.
- **F2 — joint-box collapse for zone 3 (v2.2 D13 degeneracy).**
  Zone 3's capability t = 0 in EVERY cycle: its two corridors'
  own-end terminals (buses 16 vs 14/17) are electrically adjacent, so
  the four sign vertices (moving q_13 and q_23 in OPPOSITE directions)
  are infeasible for any t > 0 — the inscribed box collapses. Zone 3
  can request but can never support. Zones 1 (t ≈ 14) and 2 (t ≈ 4)
  are unaffected. This is the anticipated cost of the v2.2 joint-box
  guarantee under collinear corridor couplings; a per-corridor
  (non-joint) capability would offer nonzero support at the price of
  losing the any-sign-combination guarantee. Record for the
  dissertation's D13 discussion.
- **F3 — voltage_margin_pu recalibrated 0.005 → 0.01** (v2.2 item 3
  verification): worst observed within-cycle supporter-side terminal
  shift over deal cycles is 6.4 mpu — one dv quantum step (~1.7 mpu)
  plus the neighbour's recovery transient, which cannot be
  observationally separated. Plan §7 evidence rule; documented in
  `SBXConfig.voltage_margin_pu`.
- **F4 — consistency counts** ((1,2): 21 no_surplus; (1,3): 5
  consistent; (2,3): 5 magnitude_off): on (2,3) the stress-driven
  natural flow shift (~+100 Mvar) dwarfs the scheduled surplus — under
  §2.5 this lands in tier 3 at the requester unless the per-line
  decomposition attributes it to ΔP/terminal-voltage deviations of the
  stressed side. Phase 6 must verify the attribution handles this
  case; the `magnitude_off` label is the early-warning marker.
- **F5 — border actuators (amendment 1):** two zone-1 AVR generators
  sit one transformer from corridor (1,2) terminals (gen 0 → bus 1,
  gen 9 → bus 38). No invariant violation or artefact observed in any
  run; corridor (1,2) executed no deals in the smoke scenario, so the
  acting-side behaviour with border AVRs remains to be watched in
  Phase 7 scenarios that exercise (1,2).

Phase 5 complete. **Next: Phase 6 (settlement).**

---

## 2026-07-07 — Phase 6: settlement ✅

### 6.1 Delivered

- `sbx/settlement.py` — §2.5 three-tier settlement per elapsed cycle on
  cycle-averaged measurements: tier 1 in-band free + unmonetised signed
  netting ledger; tier 2 paid-surplus billing
  `p_surplus × |paid| × t_cycle`, payer = the importing (non-acting)
  side (§2.5 verbatim); tier 3 beyond-band excess at `κ·p_surplus`,
  attributed by the dominant term of the per-line first-order
  decomposition (C_A | C_B | C_P from `corridor_sensitivities` at the
  elapsed references incl. the acting dv), ΔP-dominant →
  settlement-neutral, residual > max(1 Mvar, 20 %·excess) →
  `UNATTRIBUTED`, no charge. Payment conservation asserted per corridor
  and cycle. `n_settle_cycles > 1` → rolling-mean window (short-cycle
  ablation). `write_settlement_outputs` → ledger CSV + Markdown summary
  under `results/<experiment>/`.
- `sbx/scheduler.py` — `record_step` gains per-line terminal-voltage
  feeds; every boundary settles the ELAPSED cycle (schedule,
  references, paid/unpaid captured before overwriting); engines exposed
  as `settlement_engines`/`settlements`. `sbx/adapter.py` supplies the
  terminal voltages, each side from its own area's measurement.
- `tests/sbx/test_settlement.py` — acceptance per plan §4 Phase 6:
  per-tier synthetic trajectories, importer-pays both surplus signs,
  mutual-unpaid not billed, partial paid split, side-A/side-B
  attribution with κ-charge, ΔP neutrality, `UNATTRIBUTED`,
  conservation, rolling window, output files, and the scheduler-level
  paid-first-unwind visibility (billed Mvar·h returns to zero
  monotonically). `pytest tests/sbx` → **57 passed**.

### 6.2 Closed-loop verification (three-arm smoke re-run, PASSED C1–C8)

Settlement ran through all 22 cycles of the sbx arm without a
conservation failure. Totals
(`results/sbx_phase5_smoke/smoke_sbx_settlement_{ledger.csv,summary.md}`):

| Corridor | Netting [Mvar·h] | Tier-2 [EUR] | Tier-3 charged | UNATTRIBUTED | Net payments [EUR] |
|---|---|---|---|---|---|
| (1,2) | −0.784 | 0.00 | 20/21 | 0 | z1 +773.69, z2 −773.69 |
| (1,3) | −0.172 | 100.00 | 20/21 | 0 | z1 +815.80, z3 −815.80 |
| (2,3) | +0.141 | 100.00 | 20/21 | 0 | z2 +3110.37, z3 −3110.37 |

### 6.3 Finding F6 — tier 3 dominates: the band is far below the noise

Tier-3 charges fired in 20 of 21 settled cycles on EVERY corridor —
including (1,2), which never dealt — and exceed tier-2 by 8–30×. Cause:
the standing `|q_meas − q_sched|` deviation (20–60 Mvar) from mpu-scale
terminal-tracking residuals × tie stiffness (40–75 Mvar/mpu) plus the
stress-event flow redistribution; `q_band_mvar = 5` (plan §5) is far
below this noise floor, so ordinary operation is priced as
"unsolicited excess". The attribution itself behaves as designed
(0 UNATTRIBUTED; sides found via the terminal-voltage terms). Before
Phase 7's payment metrics mean anything, Manuel must decide:
recalibrate `q_band_mvar` to the realised noise (e.g. per-corridor,
~2σ of the no-deal deviation), and/or make tier 3 conditional on a
persistence criterion. Open decision D-P7-5 (HANDOVER_SBX_PHASE7.md).

Also noted (Phase 6 design record): §2.5's tier-2 payer rule
("importing side pays") inverts the requester-pays intuition for an
EXPORT-need requester — the acting side is then the requester and the
supporter would pay. Not exercised by the import-need smoke scenario;
flagged for symmetric/overvoltage scenarios (D-P7-3).

Phase 6 complete. **Next: Phase 7 (experiments) — handed over, see
`HANDOVER_SBX_PHASE7.md`.**

---

## 2026-07-07 — Phase 7: demonstration campaign (in progress, second session)

### 7.1 Reframing (Manuel, 2026-07-07 — supersedes the plan-v2 §4
### acceptance ordering)

Phase 7 is a **mechanism demonstration**, not a benchmark contest:

- **Primary acceptance = mechanism behaviour**, generalising the smoke
  criteria: need flags fire correctly under stress; unilateral deals
  relieve the requesting area (measured against `sbx_inert`, the
  pinning-only baseline); parallel same-cycle deals stay jointly
  feasible (supporters violation-free); symmetric scarcity degrades
  gracefully with `ScarcityEvent`s logged and `q_sched` settling;
  unwind returns to standard on budget; settlement conserves and the
  ledger is interpretable.
- **Metrics table:** mechanism-centric quantities (deals, scarcity,
  unwinds, exchanged |ΔQ|, relief vs `sbx_inert`, spillover per
  corridor, payments per area) with Φ and violation exposure as
  DESCRIPTIVE columns — no acceptance gate on either (resolves D-P7-1).
- **BME arm:** optional context on one or two scenarios, "for
  orientation" — BME = price-signal layer, SBX = operational contract
  layer; different design goals, not competitors on one axis. Kept
  code-isolated (Manuel: BME will likely leave the thesis/codebase).
- D-P7-2: `sbx_inert` is the published baseline arm. D-P7-3: §2.5
  payer rule kept as-is, affected cycles reported. D-P7-4: k_sched
  ablation deferred (no flag implemented yet — added when needed).

### 7.2 Deliverable: `experiments/013_SBX_LADDER.py` (new file only;
### `sbx/` and the runner untouched by this session)

- Arms per scenario: `none` / `sbx_inert` / `sbx` (+ optional `bme`
  via `--with-bme`, only on `asym_z3`; 011 constants imported inside
  the branch — no top-level BME dependency).
- **Scenario families** (smoke timing: freeze min 30, stress 60–210):
  `asym_z3` (validated smoke reference), `asym_z1` (border-actuator
  watch F5), `asym_z2` (F2: zone 3 cannot support), `sym_z1z2`
  (opposite-sign requests on (1,2) → scarcity), `compl_z1z3`
  (overvoltage z1 + undervoltage z3 → mutual/unpaid deal, exercises
  D-P7-3). New-scenario stress magnitudes locked via `--calibrate`
  (120-min sbx-only pass, calibration-horizon rule).
- **D-P7-5 handling (F6):** per-scenario tier-1 band calibration at
  config level — `q_band(sbx arm) = max(5, ceil(2 × RMS))` of the
  `sbx_inert` arm's clean-cycle `q_meas − q_sched` deviation (stress
  window + 4 unwind cycles excluded). The inert arm runs on the §5
  default band (its payment columns are not headline). No `sbx/` edit.
- Mechanism flags M1–M7 per scenario (printed and tabled, not
  process-fatal); metrics.csv (one row per scenario × arm); plots per
  plan §4 (P1 q_sched/q_meas/band + deals, P2 terminals vs refs,
  P3 surplus + need flags, P4 cumulative payments); REPORT.md;
  spillover per v2.2 item 6 = q_meas(sbx) − q_meas(sbx_inert) on
  no-deal stress cycles.

Status: script ready, all arm configs construct; calibration pass for
the four new scenarios launched. Full campaign after calibration.

### 7.3 Campaign results (2026-07-08, five scenarios × three arms × 360 min)

**Every applicable mechanism flag PASSES on every scenario** (M1–M7;
M3/M5 n-a on `sym_z1z2` — no deals execute in the scarcity
demonstration, correctly). All three deal archetypes demonstrated on
the triangle: unilateral paid relief (`asym_z1/z2/z3`), graceful
scarcity (`sym_z1z2`: 10 `ScarcityEvent`s, 0 deals, exposures within
0.08 pu·step of autonomous), mutual unpaid exchange (`compl_z1z3`:
2 mutual + 7 unilateral deals). Contract caps saturate as designed
(peak surplus exactly 50 Mvar with 5 cap-rejections on three
scenarios); zone-3 (0,0) offers produce dust rejections (8–20 per
scenario) instead of deals — F2 replicated (t_z3 ≈ 0 in every
scenario; t_z1 ≈ 14, t_z2 ≈ 1.3–4).

Headline numbers (full table: `results/013_SBX_LADDER/metrics.csv`;
plots P1–P4 + settlement ledger/summary per scenario directory;
REPORT.md with the per-scenario flag tables):

| Scenario | Deals (uni/mut) | Scarcity | Pinning cost [pu·step] | Deal benefit [pu·step] | q_band* [Mvar] | Tier-2 [EUR] | Tier-3 cycles (sbx / inert@5) |
|---|---|---|---|---|---|---|---|
| asym_z3 | 4 / 0 | 0 | +0.354 | +0.019 | 34 | 200 | 16 / 60 |
| asym_z1 | 10 / 0 | 0 | +0.381 | +0.028 | 92 | 625 | 26 / 58 |
| asym_z2 | 5 / 0 | 0 | **−0.166** | +0.014 | 26 | 750 | 19 / 53 |
| sym_z1z2 | 0 / 0 | 10 | +0.053 | 0 | 26 | 0 | 34 / 53 |
| compl_z1z3 | 7 / 2 | 0 | −0.061 | −0.029 (z1) | 83 | 850 | 16 / 59 |

Findings:

- **F1 replicated and REFINED:** pinning cost ≫ deal benefit on
  `asym_z3`/`asym_z1`, but on `asym_z2` the pinning cost is NEGATIVE
  (−0.166 pu·step — the frozen contract voltages happened to sit above
  zone 2's stressed operating point, so pinning helped) and the deals
  add further relief: exposure ordering sbx < inert < none. The
  pinning effect is a lottery of the freeze point, not a uniform tax.
- **F6/D-P7-5 quantified:** calibrated bands are strongly
  scenario-dependent (26–92 Mvar ≙ 2×RMS of 12.6–45.6 Mvar clean-cycle
  deviation). Even calibrated, stress-window cycles exceed the band
  (16–34 tier-3 cycles vs 53–60 on the default 5-Mvar inert arms).
  A contract-constant band cannot serve both quiet and stressed
  operation; per-corridor calibration plus a persistence criterion is
  the evidence-backed recommendation for the dissertation.
- **D-P7-3 evidence delivered:** in `compl_z1z3` the export-need
  requester (zone 1) RECEIVES +456 EUR — the §2.5 "importing side
  pays" letter makes the supporters pay the party whose need triggered
  the deal. Now documented with numbers for the rule discussion.
- **Spillover (v2.2 item 6) is negligible:** ≤ 0.98 Mvar max,
  ≤ 0.49 Mvar mean on no-deal corridors across all scenarios — deals
  do not leak measurably around the triangle.
- **Consistency classification:** first `sign_mismatch` labels
  appear under deep stress (8 on `asym_z1`, 3 on `asym_z2`,
  1 on `compl_z1z3`; smoke had zero): the stress-driven natural flow
  shift can oppose the scheduled surplus direction. Diagnostic only,
  as designed. `UNATTRIBUTED` fired exactly where intended: 5 cycles
  on `asym_z1` (deep-stress decomposition residual).
- Φ (descriptive): differences between arms ≤ ~2 MW mean on four
  scenarios; `asym_z1`'s Φ is band-hinge-dominated (≈ 465 MW) under
  the deep violation. No ordering claims made (demonstration framing).

Definition of done (§8): `pytest tests/sbx` **57 passed**;
`experiments/013_SBX_LADDER.py` reproduces table and plots
(`--run`/`--evaluate`); no BME path, vertical CAIR, MIQP assembly or
solver wrapper touched by this session (new files only). BME context
arm not run (optional `--with-bme`, kept code-isolated; Manuel
2026-07-07: BME likely leaves the thesis/codebase).

**Phase 7 complete.** Open for Manuel: D-P7-5 final band rule for the
dissertation text; whether the BME orientation arm is wanted at all.

### 7.4 Single-run demo + live Figure 6 (2026-07-08)

Added on Manuel's request (mechanism visualisation with the
no-remuneration band, toggleable like the other live plots):

- `visualisation/plot_sbx.py` — `SBXMechanismLivePlotter` ("Figure 6 —
  SBX MECHANISM"): per corridor the measured reference-end flow vs the
  q_sched staircase / q_std with the tier-1 band shaded, deal markers
  (▼ unilateral, ◆ mutual, △ unwind, ✕ scarcity) and need-flag strips;
  below, the surplus staircases and cumulative per-area payments. The
  window opens at run start with a placeholder and populates at the
  contract-freeze tick.
- `MultiTSOConfig.live_plot_sbx` (default False) + minimal runner
  wiring (construction beside the other plotters; per-step
  `update(rec, adapter, corridor_q)`; fail-fast without
  `coordination_mode="sbx"`); plotter handle exposed via
  `sbx_runtime["live_plotter"]`.
- `experiments/014_SBX_SINGLE_DEMO.py` — one simulation of any 013
  scenario with the live figure; per-scenario calibrated band defaults
  (34/92/26/26/83 Mvar), `--band` override, `--no-live`; saves the
  final PNG + settlement ledger/summary + corridor cycle table.
  Validated headless end-to-end on `asym_z3` (150 min).

**Finding F7 (shared-code, NOT fixed here):** `rec.zone_tie_q_mvar` /
`rec.tie_q_mvar` (`_multi_tso_helpers.py:586-604`) compute the pair
flow as −q_from when a tie line is oriented FROM the higher zone — a
proxy that ignores line charging. On line 14 (b_sh = 1.2 pu, oriented
8→38) this misstates the at-endpoint flow by ~107 Mvar. Consumers:
Figure-1 tie tile, 011 tie heat-map, zigzag diagnostics. Figure 6
computes its own reference-end flows (q_from/q_to per orientation) and
is unaffected; the records docstring ("positive = Q leaves zone i")
does not match the computation for flipped charged lines — Manuel to
decide whether to fix the helper or the docstring (BME-era consumers
may rely on the current values).

Also observed: back-to-back runs of the identical 014 configuration
differ at the ~0.4 Mvar level (Gurobi MIQP tie-breaking is not
deterministic across runs); cycle-level mechanism behaviour is
unaffected but exact deal timing can shift by a cycle near threshold.

### 7.5 Local-sensitivities restriction lifted (2026-07-08, Manuel's request)

The Phase-5 runner validation excluded `local_sensitivities_*` under
`coordination_mode="sbx"` as a conservative fail-fast on an unvalidated
path — not for a structural reason: the adapter reads only
controller-owned cached objects (the voltage rows of the controller's
own H; `ctrl.sensitivities.compute_dV_dQ_der`), and the local
(Ward-style reduced-net) `JacobianSensitivities` provides the identical
interface with original in-zone bus indices. Local cached models are in
fact the configuration most consistent with the SBX locality principle
(plan §2.4). Change: the runner now excludes only `numerical_h`;
`experiments/014_SBX_SINGLE_DEMO.py` gains `--local-sens` (and `--arm`,
supporting Manuel's baseline probing without editing the file).
Validation (asym_z3, 150 min, headless, local TSO+DSO sensitivities):
exit 0, full protocol arc (deals c3–c4 on both zone-3 corridors,
unwind c7–c8 to zero), dv per quantum identical to the shared path
(±1.68/0.80 mpu), relieving-sign assert silent, settlement complete.
013's campaign arms keep the shared path (already run; comparability).

### 7.6 Dissertation mechanism reference written (2026-07-08)

New DRAFT-marked section "Scheduled Boundary Exchange: An Operational
Contract Layer" (`ch:architectures:multitso:sbx`, 10 subsections)
inserted at the head of dissertation Chapter 7 on Manuel's request:
corridors/contract data (v_std definition + precision requirement from
tie stiffness), the π-model q(p) characteristic and q_std, the moving
band + 2×RMS calibration rule, need flag + request-sign mapping,
joint-box capability LP, message schema + checksums, matching rules,
Step-4 dv equation + acting-side invariant + pinning cost, unwind,
three-tier settlement with the attribution decomposition, a worked
asym_z3 arc, parameters/information accounting/limitations. Placement
provisional (head of chapter, to be relocated when the chapter
structure is decided). Compiles cleanly under LuaLaTeX (zero errors;
note: the thesis builds with lualatex, not pdflatex).

### 7.7 D-P7-4 ablation results (2026-07-08, asym_z3, 360 min)

| variant | cycle | quantum/cycle | exposure z3 [pu·step] | deals | exchanged [Mvar] | first deal [min] | tier-2 [EUR] |
|---|---|---|---|---|---|---|---|
| none | — | — | 1.645 | — | — | — | — |
| sbx_inert | 15 min | — | 1.999 | 0 | 0 | — | 0 |
| sbx | 15 min | 10 | 1.980 | 4 | 40 | 75 | 200 |
| sbx_fast | 3 min | 2 | 2.005 | 22 | 44 | 75 | 143 |
| sbx_bigq | 15 min | 30 | 1.962 | 4 | 120 | 75 | 600 |
| sbx_fast_bigq | 3 min | 6 | 1.984 | 20 | 120 | 75 | 360 |

Findings (F9): (i) the FIRST DEAL lands at minute 75 in every variant —
deal latency is bound by the need-flag persistence (n_need = 5 TSO
iterations = 15 min), not by the cycle length; a short cycle buys
nothing unless n_need shrinks with it. (ii) Exposure is flat across all
variants (1.96–2.00, differences within the ±0.02 solver-nondeterminism
noise) even at 3× the exchanged volume — the STOPPING RULE binds: the
flag clears once the violation dips below the 5-mpu threshold, so every
variant converges to the same "just below flag depth" state and only
the price differs (200 → 600 EUR). (iii) The mechanism levers that
would change outcomes are the need semantics (preventive/deeper
trigger: request until a comfort margin, not until flag-clear) and the
acting-side pinning — not quantum or cycle length. Fix F8 en route:
rolling-window tier-2 payer direction (sbx/settlement.py; regression
test replaces the obsolete rep1 test; 57 passed).

---

## 2026-07-08 — SBX v3 amendment: planning-anchored contract voltages
## (in progress)

Manuel approved the design proposed after the D-P7-4 ablation: the
contract voltages become an HOURLY SCHEDULE from a planning power flow
(the DACF/IDCF analogue) instead of the settled-state snapshot.

Normative changes (v3):

1. **Contract.** `CorridorContract` gains an optional
   `v_std_schedule`: an ordered tuple of `(t_from_s, v_std_a, v_std_b)`
   intervals in scenario time, first interval starting at 0, last
   extending to the horizon; `v_std_at(t_s)` resolves the active pair.
   With a schedule present, the constant `v_std_*_pu` fields must equal
   the first interval (t = 0 view); `q_std_mvar` requires `time_s`
   when a schedule exists (no silent constant fallback).
2. **Scheduler.** Cycle boundaries resolve the ACTIVE contract voltages
   once per corridor (Step 1) and use them everywhere the constant
   `v_std` was used (q_std, capability sensitivities, Step-4 refs +
   dv solve + invariant). `run_cycle`/`initial_references` take the
   scenario time; the surplus/unwind/settlement semantics are unchanged
   (an hourly v_std step is structurally the same event as the existing
   per-cycle p_sched step).
3. **Adapter/runner.** `SBXRunnerAdapter` takes `freeze_time_s` and
   optional per-corridor schedules; `MultiTSOConfig` gains
   `sbx_v_std_schedule_path` (JSON written by the pre-pass; keys
   "i-j", entries `[t_from_s, [v_a per line], [v_b per line]]`). With a
   schedule, the snapshot step is skipped — planning replaces
   measurement; the warmup keeps only its adapter-construction role.
4. **Pre-pass.** `experiments/017_SBX_PLANNING.py`: hourly planning
   power flow over the scenario horizon on the same net/profiles/zonal
   dispatch, WITHOUT contingencies and without the closed loop; modes
   `perfect` (true profiles) | `persistence` (profiles of t − 24 h) |
   `noise` (calibrated injection noise). Output: schedule JSON + plot.
5. **Rationale.** Dissolves F1's freeze-point lottery; tier 3 becomes
   "deviation from the agreed plan"; the tier-1 band becomes a
   forecast-quality statement. Dissertation §sbx:contract already
   anticipates this definition.

### v3 validation (2026-07-08) and finding F10 — planning-model fidelity

Machinery validated end-to-end (`pytest tests/sbx` 58 passed; 150-min
run with warmup + schedule; 90-min run with contracts ACTIVE FROM
t = 0 — the warmup is obsolete under a schedule and 014 sets
`sbx_warmup_s = 0` with `--schedule`): hourly v_std switches land as
q_std steps at the right boundaries, Step-4 references follow the
active interval, deals/unwind/settlement run unchanged, and under
Manuel's new aggressive defaults (6-min cycles, 12 Mvar quantum,
n_need = 1) the mechanism stacks three deals within 18 min of the
stress onset.

**F10 — the crude planning view is too crude for the stiff ties.** The
perfect-mode pre-pass (taps/shunts at build defaults, no STATCOM/OLTC
scheduling, gens at 1.03) plans corridor (1,2) at ≈ −298 Mvar where the
closed loop realises ≈ −130 Mvar — a standing ≈ 165 Mvar plan-reality
gap (a few mpu of terminal-difference error × 40–75 Mvar/mpu
stiffness); (1,3) plans ≈ +2 vs realised ≈ +30. Everything works, but
tier 3 prices the model gap nearly every cycle on (1,2). Options for
Manuel: (a) schedule the controllable states in the plan as real
planners do — add the runner's operating-point phases (machine-trafo
DiscreteTapControl + coupler OLTC + STATCOM init) to the 017 pre-pass
(recommended); (b) per-corridor/hourly bands calibrated against the
planning-error statistics (017 persistence/noise modes provide them);
(c) keep the crude plan as the honest "cost of a bad plan"
demonstration for the dissertation.

### F10 resolved (2026-07-08): scheduled operating point in the plan

Option (a) implemented: `experiments/017_SBX_PLANNING.py` now schedules
the controllable states per planning hour exactly as the runner's
operating-point phases do (TN-side STATCOMs via the temp-PV-gen trick +
machine 2W OLTC at the zone voltage schedule, then coupler 3W OLTC at
the DSO target; taps persist across hours = hourly re-scheduling).
`--no-oltc-schedule` reproduces the crude F10 plan (output tagged
`_crude`) for the "cost of a bad plan" comparison.

Closed-loop verification (asym_z3, 90 min, contracts from t = 0,
6-min cycles): corridor (1,2) plan −102 Mvar vs realised −115 (gap
≈ 165 → ≈ 13 Mvar); corridor (1,3) plan +37.16 vs realised +37.10
(sub-Mvar). The early cycles show the plant CONVERGING ONTO the plan
(c1 deviation 54 Mvar → c4 0.35 Mvar on (1,2)), and after the stress
hit the measured flow relaxes back toward the planned standard — the
plan acts as the reference through the disturbance, as intended.
Remaining residual = genuine model difference (planning view vs closed
loop), the quantity the tier-1 band should be calibrated against.

Recorded for the dissertation: the planning-fidelity ladder crude PF →
rule-based scheduled operating point (implemented) → ORPF (future
`--mode orpf` via pandapower OPF / `ac_opf_reference.py`; taps
relaxed-and-rounded). ORPF anchoring makes v_std the jointly agreed
OPTIMAL plan — the strongest justification of tier-3 pricing — and
completes the symmetry: BME computes online what ORPF-anchored SBX
fixes in planning.

### v3 closing piece (2026-07-08/09): planning-derived hourly band

Manuel's question "could q_band also be derived from the day-ahead
calculation?" answered affirmatively and implemented:

- `experiments/017_SBX_PLANNING.py --band-ensemble N` (default 8): per
  hour, N forecast-error-sampled power flows with taps HELD at the
  day-ahead plan; hourly per-corridor band = z·σ_ensemble +
  |s_corr|·ε_track (stiffness × declared tracking tolerance, contract
  data) + m_gap (backtested model allowance), floored at 5 Mvar.
  Result on the perfect-mode schedule: (1,2) 23–33, (1,3) 14–15,
  (2,3) 25–36 Mvar — within a few Mvar of the 013 empirical 2×RMS
  calibrations (independent cross-validation of D-P7-5).
- Contract: `q_band_schedule` + `q_band_at(t)` (same interval rules as
  v_std); scheduler resolves the ACTIVE band per cycle and settles the
  ELAPSED cycle with the band that governed it; settlement windows the
  band like every other quantity; Figure 6 shades the per-cycle band
  (visibly breathing at hour boundaries); `CorridorCycleRecord` and
  `CycleObservation` carry `q_band_mvar`.
- 014 defaults changed on Manuel's expectation: planning schedule
  auto-detected (contracts from t = 0; `--no-schedule` for the v2
  snapshot) and LOCAL sensitivities by default (`--shared-sens` opts
  out). End-to-end validated (90 min, asym_z3): plan-reality gap as in
  the F10 fix, all bands active, payload pickle persisted.
  `pytest tests/sbx` → 58 passed.

### OPEN (2026-07-09): "v4 — deliverable SBX" architecture decision

Manuel's aggressive-defaults run (n_need = 1, cap 150, 6-min cycles)
exposed the delivery gap: 150 Mvar scheduled on (1,2) while the
measured flow diverged by ~250 Mvar — w_track ≡ g_v gives the contract
reference no priority, the joint-box offer never becomes a controller
instruction, the requester is pinned, and Q-locality caps the reach.
Proposed package (analysis in the session log, awaiting Manuel's
decision): (i) per-bus tracking weights (w_track ≫ g_v on corridor
terminals — TSO-controller gradient change, previously off-limits for
BME-era protection), (iii) delivery-conditioned requesting (pause on
sign_mismatch — reuses the consistency classifier), (A) per-corridor
capability with the cross-corridor externality priced by the EXISTING
tier-3 attribution (replaces D13's any-combination guarantee, which
collapses for zone 3's collinear geometry — F2).

---

## 2026-07-09 — SBX v4 "deliverable SBX" (implemented; validation
## running)

Manuel approved the full package (i) + (iii) + (A) after the delivery-
gap analysis (150 Mvar scheduled vs ~250 Mvar divergence under the
aggressive defaults):

1. **(i) Per-bus tracking weights — the contract gets priority.**
   `TSOControllerConfig.g_v_per_bus` (optional vector aligned with
   `voltage_bus_indices`; None = scalar `g_v`, bit-identical legacy) +
   `TSOController._g_v_vector()` used at all three objective sites
   (output-gradient, grad_f, curvature view) +
   `update_voltage_tracking_weights(bus_indices, weight)`.  First
   deliberate TSO-controller change of the SBX effort — the A4/G5
   protection existed for BME-era code and is lifted.  The adapter
   applies `w_track` (absolute) or `w_track_factor × g_v` (new
   `SBXConfig.w_track_factor = 20`) to every corridor terminal at
   construction; the old w_track ≡ g_v assertion is gone.
2. **(iii) Delivery-conditioned requesting (`delivery_gate = True`).**
   While a corridor's last settled cycle is `sign_mismatch` /
   `magnitude_off`, new requests are suppressed
   (`CorridorCycleRecord.request_suppressed_a/b`) and the non-delivery
   counts like a cleared flag towards the unwind dwell — undelivered
   support is wound back instead of billed.  Emergent property
   (documented + tested): after a full unwind the evidence resets and
   the mechanism PROBES again — under persistent non-delivery a
   bounded deal/mismatch/unwind retry loop (surplus ≤ ~1 quantum)
   replaces the pre-v4 ratchet to the cap.
3. **(A) Capability modes (`capability_mode = "auto"`).**
   `sbx.capability.area_capability` dispatcher: D13 joint box first;
   when its offers all fall below the dust threshold (the F2 collapse
   on collinear corridor couplings), per-corridor 2-vertex LPs take
   over — own corridor guaranteed, cross-corridor side effects
   tolerated and PRICED by the existing §2.5 tier-3 attribution
   (geometric prohibition → economic accountability).
   `CapabilityResult.mode` records which path served the offers.
   Zone 3 can now support.

Tests: three legacy harness tests pin `delivery_gate=False` (their
synthetic tie_q = 0 feed is exactly the non-delivery the gate acts on);
new `tests/sbx/test_v4_deliverable.py` covers the gate (suppression,
bounded probing, non-delivery unwind, honest need flag), the F2
collapse + auto fallback + joint-box preservation, and the per-bus
weight validation.  `pytest tests/sbx` → **63 passed**.

### v4 validation (2026-07-09, asym_z3, 90 min, all v4 defaults)

The protocol arc on corridor (1,3) is the v4 result in miniature:
stress at minute 60 → need fires → ONE unilateral deal (+12 Mvar,
dv = +2.0 mpu on the priority-tracked acting terminals) → the next
boundary classifies the elapsed cycle **"consistent"** — the realised
corridor deviation matched the scheduled surplus in sign and magnitude
(pre-v4 the same configuration produced sign_mismatch chains and a
150-Mvar ratchet) → the delivered quantum plus zone 3's own actuators
clear the violation within one cycle → dwell → unwind → both ends back
on the hourly contract voltages. One deal, delivered, settled, released
— in four cycles. The delivery gate never had to fire (nothing was
undelivered), no ratchet, corridor (1,2) stayed inside its hourly band
throughout. `pytest tests/sbx` 63 passed.

Note for the thesis: all pre-v4 characterisations (F1 pinning/benefit
numbers, F9 ablation, 013 campaign tables) describe the v2/v3
mechanism; re-run 013 under v4 before citing final numbers.

### 2026-07-09 (later): per-corridor default, modal evidence, 180-min
### evaluation, classifier fix

- **capability_mode default → "per_corridor"** (Manuel): the outer
  approximation is the vertical-CAIR philosophy per interface; for
  small quanta the over-promise is bounded and policed (margin + gate +
  attribution). Scheduler warns when Σ quantum/|s_corr| exceeds the
  margin budget (the quantifiable small-quantum condition).
- **Modal evidence recording** (Manuel's z± = (Δq1 ± Δq2)/2 proposal):
  `sbx.capability.modal_capability` (two support-function LPs) records
  (a_plus, a_minus) per two-corridor area per cycle in
  `SBXScheduler.modal_records` (config `record_modal_capability`).
  Diagnostic only; the v5 modal-offer protocol stays a documented
  candidate.
- **180-min asym_z2 evaluation (F11):** the two supporters split
  exactly along the modal prediction — zone 1 (separated corridors):
  13 deals to the 144-Mvar cap at delivery ratio ≈ 0.86 with 12
  'consistent' cycles; zone 3 (collinear, per-corridor over-promise):
  12 deals at delivery ratio ≈ −0.30, zero 'consistent', gate fired 3×.
  Differential-mode support is physically undeliverable for zone 3
  (a_minus ≈ 0 in action) — the empirical case for the modal offer.
- **Classifier blind spot fixed:** 'deadband' may short-circuit only
  when the scheduled surplus itself fits inside the band; previously a
  108-Mvar phantom surplus rode the 26-Mvar band as 'deadband' and
  starved the delivery gate.
- **Test-suite pinning:** protocol tests now use `ref_config()`
  (tests/sbx/test_scheduler.py REF_CFG) — SBXConfig defaults are
  Manuel's live experimental knobs (changed three times today) and no
  longer break tests. 133 passed (sbx 100 + sbxv 33).

---

## 2026-07-10 — Package rename (sbx → sbx_h) + 015 helpfulness evaluation

### Rename

`sbx/` → `sbx_h/` and `sbxv/` → `sbx_v/` (tests folders alike:
`tests/sbx_h`, `tests/sbx_v`), on Manuel's request — the unqualified
name became ambiguous once the vertical mechanism existed. Hyphenated
names ("sbx-h") are not valid Python package names; underscores used.
All imports and dotted references rewritten; the runner now accepts
`coordination_mode = "sbx_h"` / `"sbx_v"` as ALIASES while the internal
mode strings stay `"sbx"` / `"sbxv"` (configs, pickles and tests
predate the rename); `MultiTSOConfig` field names unchanged
(`sbx_config`, `sbx_warmup_s`, …). Historical entries in this file keep
the old paths (they document the past). Verification: full suite
`pytest tests/sbx_h tests/sbx_v` → **165 passed**; import smoke of all
SBX experiments green. Details:
`docs/daily_log/07_2026/2026-07-10_sbx_h_rename_and_015_helpfulness.md`.

### 015 (rewritten) — when is SBX-H useful over NO explicit communication?

Manuel's standing question, now formalised and tested by construction.
Helpfulness conditions (all necessary): C1 persistent violation in A
with A's own actuators EXHAUSTED (gen/DER reserves ≈ 0 in the relieving
direction); C2 corridor controllability of the violated buses; C3
supporter capability and v4 deliverability (t ≥ 1, delivery ratio ≈ 1);
C4 counterfactual gap (passive support + autonomous tracking do not
already close it). Bounds on the achievable benefit: F9 stopping rule
(steady state "just below flag depth") and the contract cap.

Design: 2×2 matrix (D = zone-3 deficit 500 vs 150 Mvar @ bus 15,
z3 v_min = 1.00, stress min 60 → horizon end; S = supporter headroom,
S0 adds 500 Mvar @ bus 27 (z1) + 450 Mvar @ bus 6 (z2) to collapse
t_z1/t_z2 without violating their bounds) × arms none / sbx_inert /
sbx (v4; knobs pinned: k_sched = 2, quantum rate 30, n_need = 2; band
per cell = 2×RMS of inert pre-stress deviation). Decomposition:
pinning cost = inert − none, deal benefit = inert − sbx, NET value vs
no communication = none − sbx (all zone-3 violation exposure). Flags
H1–H6 encode "helpful exactly iff deficit ∧ headroom". The old
2026-07-08 figure-only 015 is absorbed (FIG_B/FIG_C) and in git
history. 120-min matrix per the calibration-horizon rule;
`--case-study` runs D1S1 at 360 min with stress 60–300.

### 015 results (120-min matrix, 6 cells x 3 arms + 2 D2S1 variants; flags 33 PASS / 1 FAIL)

Zone-3 violation exposure [pu*step] over the stress window (60 -> 120 min):

| cell | none | sbx_inert | sbx | pinning value | deal benefit | deals | z3 need cyc. |
|---|---|---|---|---|---|---|---|
| D2S1 exhausted + headroom | 5.207 | 1.444 | 1.442 | +3.763 | +0.001 | 2 (gated) | 6-9 persistent |
| D2S0 exhausted, margin-zero supporters | 5.207 | 1.444 | 1.399 | +3.763 | +0.045 | 2, no stacking | 6-9 persistent |
| D1S1 misdirected + headroom | 1.458 | 0.119 | 0.115 | +1.339 | +0.004 | 2, unwound | 6 only |
| D1S0 misdirected, margin-zero supporters | 1.458 | 0.119 | 0.115 | +1.339 | +0.004 | 2, no stacking | 6 only |
| D0S1 / D0S0 self-sufficient | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | none |

Findings G1-G6 (full text embedded in `results/015_SBX_COMPARE/REPORT.md`):

- **G1 — three deficit regimes.** D1 (500 Mvar) is a regime the hypothesis
  missed: the violation persists under `none` NOT for lack of actuators
  (z3 reserve floor 0.23-0.40) but because uniform 1.03-tracking
  misdirects them; priority pinning redirects zone 3's OWN reserves to
  the boundary and nearly clears it. True C1 exhaustion needs 900 Mvar
  (D2): violation persists under pinning, DER reserve floor 0.013 (H7).
- **G2 — the CONTRACT layer carries the value; it needs no runtime
  communication.** Net vs `none`: +3.76 (D2S1) / +1.34 (D1S1) pu*step,
  of which deals contribute +0.001 / +0.004. INVERTS pre-v4 F1: with
  w_track_factor = 20 the pinning IS the control action.
- **G3 — the deal layer self-suppresses under deep stress,
  band-independently.** One quantum per corridor executes, then the
  delivery gate fires: the consistency classifier compares the realised
  corridor deviation (natural stress shift, order 100 Mvar; `none` arm
  pulls +310 Mvar into z3) against the 12-Mvar scheduled surplus ->
  `magnitude_off` at band 5 AND at band 35 (variant D2S1_band35
  identical). Deal delivery is unattributable against the natural flow
  redistribution with the magnitude check — the F6/D-P7-5 root cause
  now binds the CONTROL side. v5 candidate: gate on the settlement's
  per-line attribution (C_A/C_B/C_P) instead.
- **G4 — ungated deals help a little, at poor delivery** (variant
  D2S1_nogate, delivery_gate=False): 8 deals stack to 48 Mvar/corridor,
  exposure 1.278 vs inert 1.444 (deal benefit +0.166, ~11 %), realised
  extra import +11.4 Mvar -> delivery ratio 0.16, clean unwind. Real but
  an order of magnitude below the pinning value, and settlement would
  price 96 Mvar scheduled for ~11 Mvar delivered.
- **G5 — supporter capability is a continuum.** Sinks do not collapse
  it (500/450 Mvar left t ~ 23/47; voltage slack + DSO Q_PCC intervals
  dominate); margin-zero bounds cut it 4-5x (t 6.8/8.9), yet one
  quantum clears while t >= 1. With the gate capping every cell at
  <= 1 quantum, the S axis is outcome-invisible in this campaign.
- **G6 — the answer.** SBX-H is strongly useful against no
  coordination exactly in the constructed corner (persistent
  boundary-near deficit) — but through the scheduled-boundary-voltage
  CONTRACT, not the runtime exchange. Against `sbx_inert` (contract,
  zero runtime communication) the explicit communication currently buys
  ~nothing that survives the delivery-attribution problem; its honest
  ceiling in the ideal cell is +0.17 pu*step at 16 % delivery (ungated).

The single FAIL is H3 in D2S1 (deal benefit +0.001 < 0.10) — recorded
deliberately: it IS the verdict on the gated deal layer in the cell
constructed to favour it. Caveats in REPORT.md (single scenario family,
snapshot contracts, pinned knobs, plant-wide exposure metric, Gurobi
+-0.02 noise, violated region electrically close to the corridor
terminals). Outputs: `results/015_SBX_COMPARE/{matrix.csv,
FIG_A_matrix.png, REPORT.md}`, per-cell FIG_B/FIG_C + settlement
ledgers, variants `D2S1_band35/`, `D2S1_nogate/`.

---

## 2026-07-10 (later) — SBX-H v5 "evidence-based SBX" (Manuel-approved redesign)

Manuel approved the full G1–G6-derived package ("implement it"); the
v4 baseline outputs are preserved under
`results/015_SBX_COMPARE/v4_baseline/`.

### What changed (all defaults; v4 semantics remain available per knob)

1. **Move 1 — C1 arming** (`require_exhaustion_to_request = True`): a
   need flag emits a request only when the area cannot help itself.
   Two arming paths, OR-combined:
   (a) optimistic cached-model bound — self-help lift
   Σ_j |H[worst, j]| · relieving headroom_j < `c1_arming_factor` ×
   depth (integer actuators arrive frozen).  **Found insufficient on
   real controllers**: the bound counts SETPOINT headroom (AVR
   voltage boxes) that maps to no physical Q on Q-saturated machines
   — the first v5 D2S1 run never armed (c1_unarmed on every flagged
   cycle) although the DER reserve floor was 0.013;
   (b) **measured stall** (`c1_stall_cycles = 2`,
   `c1_stall_improvement = 0.3`): a flag persisting beyond the dwell
   while the violation depth recovered by < 30 % of its onset value is
   exhaustion in the behavioural sense, whatever the model bound says.
   Model-free, immune to the setpoint-vs-physical mismatch.
2. **Move 2 — voltage-referenced delivery**
   (`delivery_check = "voltage"`, `v_delivery_tol_pu = 2.5 mpu`): the
   gate verifies the ACTING side's terminal tracking against its
   shifted references at the elapsed cycle's last sample — what the
   supporter actually controls and measures locally; immune to the
   natural stress flow shift that blinds the magnitude test (G3,
   band-independent).  The magnitude classifier remains as a
   diagnostic and as the `"magnitude"` legacy mode.  **Settlement:**
   `tier2_requires_delivery = True` — paid surplus is billed only for
   delivery-verified cycles (`CycleSettlement.delivered_frac`); no
   payment for phantom support.  Tolerance caveat (documented): per-
   quantum dv on the stiff ties (~0.2 mpu) is below any realistic
   tolerance, so single quanta verify trivially; the gate's guarantee
   is BOUNDED unverifiable exposure (~tol × stiffness per corridor) —
   the ratchet, not the single quantum, is what it catches.
3. **Move 3 — preventive release + sized requests**:
   `release_threshold_pu` (need-flag hysteresis: set at 5 mpu, clear
   below the release value — removes the F9 'just below flag depth'
   stall; None = v4) and `request_sizing = "gap"` — requests sized
   ceil((depth − release)/(|dV/dQ| · quantum)) quanta, capped at
   `k_max_quanta_per_request = 4`; matching accepts integer multiples
   of the quantum; capability offers scale to min(t, k_max) quanta;
   the scheduler's margin warning accounts for k_max.
4. **Pruning:** `record_modal_capability` default → False (diagnostic
   LPs off); the magnitude test demoted to diagnostics.

Files: `sbx_h/config.py` (9 new validated knobs), `sbx_h/need.py`
(latched hysteresis; release = set reproduces v4 exactly),
`sbx_h/scheduler.py` (arming incl. stall state, sized requests,
voltage delivery verification per elapsed cycle, records gain
`c1_unarmed_a/b`, `delivered`, `v_track_err_pu`),
`sbx_h/matching.py` (multiple-of-quantum guard),
`sbx_h/capability.py` (`max_quanta` offer scaling),
`sbx_h/settlement.py` (`CycleObservation.delivered`,
delivery-gated tier 2, `delivered_frac`).

Tests: v4 protocol semantics PINNED in `tests/sbx_h/test_scheduler.py
REF_CFG` (the legacy suite keeps testing v4); new
`tests/sbx_h/test_v5_redesign.py` (10 tests: config validation,
hysteresis + v4-equivalence, matching multiples, bound-arming both
ways, stall-arming override, gap sizing to k_max, voltage gate
suppress/track paths incl. tier-2 suspension and the bounded probe
loop).  Suite: **74 sbx_h + 102 sbx_v passed**.

### v5 closed-loop results (015 matrix re-run 2026-07-12, sbx arms only; flags 32 PASS / 0 FAIL)

| cell | none | sbx_inert | sbx (v5) | deals v5 (v4) | unarmed need cyc. |
|---|---|---|---|---|---|
| D2S1 | 5.207 | 1.444 | 1.444 | 0 (2, gated) | 10 (both corridors × 5) |
| D2S0 | 5.207 | 1.444 | 1.444 | 0 (2) | 10 |
| D1S1 | 1.458 | 0.119 | 0.119 | 0 (2, unwound) | 4 |
| D1S0 | 1.458 | 0.119 | 0.119 | 0 (2) | 4 |
| D0S1/D0S0 | 0.000 | 0.000 | 0.000 | 0 (0) | 0 |

**G7 — the plant is never behaviourally exhausted; the deal layer is
correctly dormant.** Even at 900 Mvar the pinned system recovers
ALONE: worst zone-3 monitored depth 20.1 → 12.3 → 7.1 → 2.0 → 0.6 mpu
over cycles 6–10 — 65 % recovered by the third flagged boundary, so
the measured-stall clause correctly declines to arm (and the model
bound never arms: AVR setpoint headroom overstates deliverable Q of
saturated machines — the reason the stall clause exists).  All v4
dead activity is eliminated: 0 executed deals anywhere, exposures
unchanged to the ±0.02 noise, D0 non-inferiority exact.

Consequence (thesis statement): in this system class the deal layer
is an ACCELERATION option (ceiling ≈ +0.17 pu·step at delivery ratio
0.16, measured ungated under v4), never a feasibility necessity;
SBX-H's measured value is the CONTRACT tier (scheduled boundary
voltages + priority tracking; +3.76 / +1.34 pu·step vs none).  Under
v5 the deal layer is a correctly-gated escalation path whose
activation is unit-proven (stall-arming test) but out of reach for
the 005 scenario family.  Open for Manuel: (i) whether a genuinely
stuck scenario (e.g. permanent loss of zone-3 in-feed beyond the DSO
capability, or a v_min raised further) should be constructed to
demonstrate armed v5 deals end-to-end; (ii) the deliberately-unfixed
acceleration variant (arm on "deep + persistent" rather than stall)
if faster-than-autonomous recovery is wanted as a product feature.

---

## 2026-07-12 — SBX-H v6: deal layer removed; planned support added

Manuel's decision after the G1-G7 evidence and the architecture
discussion (docs/SBX_H_V6_ARCHITECTURE_CANDIDATES.md): "clean sbx-h up
to only have the necessary mechanism left", with support agreed IN
ADVANCE as a schedule product.

**v6 package = contract layer + attributed settlement + escalation
indicator.** Removed: capability/matching/messages modules, scheduler
Steps 2-5 (requests/offers/matching/dv/unwind/gates/arming),
corridor_solve_dv, settlement tier 2, all deal-layer knobs. Kept:
corridor registry + pi-line model, CorridorContract (v_std/q_band
schedules), q_std, priority terminal tracking, violation indicator
(hysteresis), tier-1 netting + attributed deviation tier (C_A/C_B/C_P,
causer-pays at kappa*p_dev_eur_per_mvarh) — the A1 remuneration hook.
Added: `with_planned_support` (a side holds its corridor terminals
shifted by dv during an agreed window; composes with planning
schedules; `MultiTSOConfig.sbx_support_intervals`) and the A4
escalation indicator (`escalation_cycles`; persistent violation or
beyond-band exceedance -> re-planning signal, recorded only).
Figure 6 adapted (q_std +/- band, deviation staircase, escalation
markers). Archive: `_archive/sbx_h_v5/` (complete v5 code + tests);
result baselines `results/015_SBX_COMPARE/{v4_baseline,v5_baseline}/`.

Suite: **35 (sbx_h v6) + 102 (sbx_v) passed** (deal-era tests
archived). 015 rewritten: cells D2/D1/D0 x arms none / sbx (contract)
/ sbx_support (+2.5 mpu on the supporters' corridor sides during the
stress window); flags V1-V6.

### 015 v6 results (2026-07-12, 3 cells x 3 arms x 120 min; 18/18 flags PASS)

Zone-3 violation exposure [pu*step] over the stress window (60-120 min):

| cell | none | sbx (contract) | sbx_support | contract value | planned-support benefit | escalation |
|---|---|---|---|---|---|---|
| D2 (900 Mvar) | 5.207 | 1.444 | 1.338 | +3.763 | **+0.106** | cycle 10 |
| D1 (500 Mvar) | 1.458 | 0.119 | 0.099 | +1.339 | +0.020 | none |
| D0 (150 Mvar) | 0.000 | 0.000 | 0.000 | 0 | 0 | none |

**The contract layer carries the value** (V2: +3.76 / +1.34 pu*step vs
no coordination; D0 non-inferiority exact) and **planned support now
delivers a clean, positive, FULLY-DELIVERED benefit** (V3: +0.106
pu*step in D2 = ~7 % of the contract arm's residual, with +9 Mvar more
tie import; +0.020 in D1). This is the decisive contrast with the
removed deal layer: where runtime deals gave +0.001 gated / +0.166 at
16 % delivery (G3/G4), planned support agreed IN ADVANCE gives full
delivery because the supporter simply tracks a RAISED reference — the
one thing the contract tier does reliably. The A4 escalation indicator
(V4) fires exactly where a re-planning signal belongs: cycle 10 in D2
(persistent indicator past 4 boundaries), nowhere in D1/D0. All solves
optimal (V5); 42 settled corridor-cycles per cell, conservation clean
(V6). Outputs: results/015_SBX_COMPARE/{matrix_v6.csv, FIG_A_v6.png,
REPORT_v6.md} + per-cell FIG_B_mechanism.png + settlement ledgers.

**Verdict for the thesis.** SBX-H reduced to its evidence-supported
core is a two-part mechanism: (1) an operable CONTRACT layer
(scheduled boundary voltages, priority-tracked, ex-post attributed
causer-pays settlement) that carries essentially all the physical
value and costs zero when idle; (2) PLANNED SUPPORT as a planning-time
schedule product (a neighbour agrees a raised boundary voltage over an
anticipated window) that adds a real, verifiable, fully-delivered
increment on top. No runtime negotiation anywhere. The removed deal
layer is documented as a negative result (unverifiable commanded
quanta on stiff AC ties).
