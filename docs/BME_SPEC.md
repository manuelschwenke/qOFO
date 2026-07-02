# Boundary Marginal Exchange (BME) — Multi-TSO Coordination for OFO-MIQP
**Design specification and build plan. This file is the driving spec for a Claude Code session.**
*(Saved verbatim into the repository on 2026-07-02 so follow-up sessions have it; originally supplied by Manuel as the session prompt. Status tracking lives in `docs/BME_STATUS.md`; session handover in `docs/BME_HANDOVER.md`.)*

How to use this document:
1. Read it fully before touching any code.
2. Work strictly phase by phase (§5). Do not start phase N+1 before phase N acceptance criteria are met.
3. Maintain `BME_STATUS.md` in the repository root (or `docs/`, matching existing convention): one section per phase, status markers ✅ / 🚧 / ❌, open questions, and the component-mapping table from Phase 0.
4. Items marked **DECISION** in §7 must be surfaced to Manuel before implementation — never decided silently.
5. All conventions in §8 are binding (Fail-Fast, British English, no silent defaults).
---
## 1. Purpose and scope
**What this adds.** A horizontal TSO–TSO coordination mechanism for the existing cascaded multi-area OFO-MIQP controller. Each TSO area keeps its local per-step MIQP (integers included: OLTC taps, MSR/MSC stages, hysteresis quantiser, dwell logic, CAIR restriction — all unchanged). Coordination enters through exactly two channels:
1. A **common objective** Φ = Σ_i Φ_i (total network losses plus soft voltage-band penalties), replacing/augmenting private per-area objectives when coordination is enabled.
2. **Boundary marginal signals** μ_j = dΦ_j/dv_b exchanged between adjacent TSOs each control step, entering the neighbour's MIQP as a linear price term on the gradient. The integers never leave the local solve.
**What this must NOT touch.**
- The vertical TSO–DSO cascade: CAIR, DSO-level OFO, the DSO feedforward correction for MSR switching — behaviour must be bit-for-bit unchanged when `coordination_mode = "none"`.
- The existing V1–V5 experiment configurations and results.
- The implemented TSO–TSO v_ref cooperative scheme: it remains available as `coordination_mode = "vref"` and is expected to be a special case of BME (§3.7); do not delete or refactor it beyond wiring it into the mode switch.
- Solver choice and MIQP formulation style (this spec only adds one linear term to the objective and one acceptance rule after the solve).
**Non-goals (out of scope, note as future work in status file):** vertical propagation of switch notices from DSO level to neighbouring TSOs; angle-based boundary coupling; online estimation of cross-area sensitivities (tie-in noted in §3.9 but not implemented here).
---
## 2. Architectural placement

```
                        ┌────────────────────────────────────────────┐
                        │            CoordinationBus (NEW)           │
                        │  MarginalSignal / SwitchNotice, delay d,   │
                        │  optional drop probability, pub/sub        │
                        └───────▲──────────────▲──────────────▲──────┘
                                │ μ, notices   │              │
   ┌────────────────────┐  ┌────┴───────────┐  ┌──────────────┴─────┐
   │  TSO area 1        │  │  TSO area 2    │  │  TSO area 3        │
   │  OFO-MIQP          │  │  OFO-MIQP      │  │  OFO-MIQP          │
   │  + BME price term  │  │  + BME term    │  │  + BME term        │
   │  + ε-acceptance    │  │  ...           │  │  ...               │
   │  + discrete slot   │  │                │  │                    │
   └───────┬────────────┘  └───────┬────────┘  └─────────┬──────────┘
           │ CAIR (existing, unchanged)                  │
   ┌───────▼────────────┐  ┌───────▼────────┐  ┌─────────▼──────────┐
   │  STS/DSO areas     │  │  STS/DSO areas │  │  STS/DSO areas     │
   │  (unchanged)       │  │  (unchanged)   │  │  (unchanged)       │
   └────────────────────┘  └────────────────┘  └────────────────────┘
   Shared:  Plant protocol (PandapowerStaticPlant / PowerFactoryPlant, unchanged)
   NEW:     BoundaryTopology, CommonObjective, MarginalComputer, CoordinationBus,
            DiscreteHygiene (slotting + ε-acceptance + horizontal feedforward notices)
```

The horizontal axis is purely **additive**: one extra term in each TSO's gradient assembly, one message bus, one post-solve acceptance check, one scheduling constraint on discrete commits. Everything else is reuse.
Component names above are **abstract**. Phase 0 maps them onto the actual repository (existing multi-area module, controller classes, sensitivity provider, config system) before any code is written.
---
## 3. Formal design
Notation follows the dissertation conventions: superscripts are value qualifiers (`meas`, `set`, `ref`, `min`, `max`, `pred`), subscripts are quantity names/indices. Plain-text math below; every new module must carry a header comment mapping code symbols to the symbols of this section.
### 3.1 Setting
- TSO areas i ∈ A = {1, …, N_A} (IEEE 39 modified, N_A = 3) with input vectors
  `u_i = (u_i_c, u_i_d)` ∈ R^{n_i_c} × Z^{n_i_d} (continuous: generator/compensator setpoints; discrete: OLTC taps, MSR/MSC stages).
- Boundary bus set B ⊂ N: all buses incident to tie-lines between different TSO areas, with per-pair subsets B_ij. A single global **boundary registry** fixes the ordering of B once; all boundary-indexed vectors use this ordering.
- `v_b^meas(k)` ∈ R^{|B|}: measured boundary voltage magnitudes at step k. Magnitudes only (reactive-power/voltage dominance assumption; angles are a noted limitation, see §7).
### 3.2 Separator assumption
**Assumption (vertex separator).** Removing the boundary buses B from the network graph disconnects the TSO areas' internal buses from one another.
Consequence used throughout: with all boundary complex voltages held fixed, each area's internal power-flow subproblem is determined by its own inputs and internal injections alone. Hence for j ≠ i, u_i influences Φ_j **only through v_b**.
This must be **asserted, not assumed**: Phase 1 implements a graph check (networkx connected-components after removing B) that raises on violation. If the current IEEE 39 partition violates it (e.g. a tie-line whose both terminals were assigned to area interiors), the fix is to enlarge B, not to weaken the check.
### 3.3 Common objective and ownership convention

```
Φ(v) = Σ_i Φ_i(v)
Φ_i(v) = w_loss · P_i^loss(v)  +  Σ_{n ∈ N_i^own} φ_band(v_n)
φ_band(v) = w_band · ( [max(0, v − v^soft,max)]² + [max(0, v^soft,min − v)]² )
```

- `P_i^loss`: active losses over branches **owned** by area i.
- `N_i^own`: buses owned by area i (internal buses plus its share of boundary buses).
- **Ownership convention** for tie-lines and boundary buses: every branch and every bus is assigned to exactly one owner, or split with fixed fractions (default: tie-line losses split 50/50, each boundary bus's band penalty owned by exactly one adjacent area — **DECISION D1**). The purpose is purely to make Σ_i Φ_i a partition of the global objective.
- **Correctness invariant (unit test, Phase 2):** Σ_i Φ_i(v) == Φ_global(v) computed centrally on the full network, to numerical tolerance, for randomised operating points.
- φ_band is a quadratic hinge: C¹ but with discontinuous second derivative at the soft-band edges. Acceptable for a gradient scheme; note it wherever a Hessian Q_Φ is assembled (§3.10) — evaluate Q_Φ at the nominal operating point and state that it is piecewise.
- The weights w_loss, w_band are **within** the single common objective, agreed once for all areas. This is not the arbitrary inter-TSO objective weighting criticised by Phulpin et al. (2009) — there is no exchange rate between heterogeneous private goals, because there are no private goals in the coordinated objective. Record this distinction in the status file; it is a positioning point for the chapter.
### 3.4 Marginal signals
For every area j (including for its own use), define the **boundary marginal**:

```
μ_j(k) = dΦ_j / dv_b  evaluated at the measured state at step k     ∈ R^{|B|}
```

where the total derivative folds in area j's internal response:

```
μ_j = (∂v_int,j/∂v_b)ᵀ · ∇_{v_int,j} Φ_j  +  ∂Φ_j/∂v_b
```

Properties:
- **Locally computable.** Under §3.2, `∂v_int,j/∂v_b` follows from area j's internal reduced Jacobian with its adjacent boundary buses treated as ports (Schur-complement of the area block). No neighbour model, no internal neighbour measurements required.
- **Sparsity.** Entries of μ_j at boundary buses not adjacent to area j are exactly zero (fixing all boundary voltages, Φ_j depends only on its own ports). Assert this sparsity pattern in tests.
- **Exchanged signal.** Each TSO j publishes μ_j(k) on the CoordinationBus. Nothing else crosses the border in steady operation: no models, no internal measurements, no objectives — a vector of |B_j-adjacent| floats per step.
- **Filtering.** Receivers apply a first-order low-pass: `μ_j^filt(k) = (1 − β)·μ_j^filt(k−1) + β·μ_j(k)`, β ∈ (0, 1] (**DECISION D3**). Purpose: a neighbour's discrete switch must not inject a step into my price signal and trigger a counter-switch.
### 3.5 Augmented per-step MIQP
Existing local per-step problem (abstract form — Phase 0 confirms the exact implemented shape):

```
Δu_i^set(k) = argmin_{Δu ∈ U_i(k)}   ½ Δuᵀ G_w,i Δu  +  α · g_i(k)ᵀ Δu
U_i(k):  CAIR restriction, actuator boxes, ramp limits,
         hysteresis quantiser and dwell constraints on discrete components
```

BME changes exactly one thing — the gradient assembly:

```
g_i^bme(k) = g_i^own(k)  +  H_{b,i}ᵀ · [ Σ_{j ∈ J} μ_j^filt(k − d) ]
```

- `H_{b,i} = ∂v_b/∂u_i` ∈ R^{|B| × n_i}: sensitivity of **all** boundary voltages to area-i inputs, columns for continuous **and discrete** actuators (the three-winding transformer tap sensitivities supply the ∂v/∂tap columns directly). In simulation this is a row/column selection of the existing global sensitivity matrix H (§3.9).
- d ≥ 1: communication delay in control steps (**DECISION D4**, default 1).
- The set J and the meaning of `g_i^own` depend on the **gradient convention audit** below. Note: under Convention A the self-marginal μ_i is computed locally and therefore enters **undelayed and unfiltered** (d = 0, β = 1 for the self term); delay and filtering apply only to signals that actually cross the border.
**Double-counting hazard (critical — Phase 0 audit item A1).** Two algebraically identical conventions exist; mixing them produces a bug that is wrong by exactly μ_i:
- **Convention A (frozen-boundary own gradient).**
  `g_i^own = ∂Φ_i/∂u_i` evaluated with v_b held fixed (area-internal reduced Jacobian only), and J = **all** areas including i itself. The self-marginal μ_i enters through the same price term as the neighbours'.
- **Convention B (total own gradient).**
  `g_i^own = dΦ_i/du_i` through all network paths (full-network H, as the current single-area OFO most likely already computes it), and J = **neighbours only**, j ≠ i.
A and B are equal by the chain rule; the audit determines which one the existing gradient assembly corresponds to and applies the matching J. Then two identity tests pin correctness (Phase 4):
1. **Single-area identity:** on a one-area network, `coordination_mode="bme"` must reproduce `"none"` exactly.
2. **Distributed = centralised gradient:** on the 3-area network, the stacked (g_1^bme, g_2^bme, g_3^bme) with d = 0 and β = 1 must equal the finite-difference gradient of the global Φ with respect to the stacked u, at randomised operating points, to tolerance.
Test 2 is the money test of the whole design. If it fails, stop and fix before proceeding.
### 3.6 Interpretation: integer externality pricing
Because H_{b,i} carries columns for the discrete actuators, the MIQP natively weighs whether a tap or shunt step is worth its externality on the neighbours — the price term converts μ into a per-actuator linear cost, integers included. Relax-round-resolve schemes structurally cannot do this (they price the relaxation, not the committed integer move). Record this as a contribution claim in the status file for later use in the chapter.
### 3.7 Relation to the implemented v_ref scheme
If area j's objective contains a quadratic boundary term `½·(v_b − v_b^ref,j)ᵀ Q_b,j (v_b − v_b^ref,j)`, then `μ_j = Q_b,j·(v_b^meas − v_b^ref,j)` — a scaled reference deviation. **Hypothesis:** the implemented cooperative v_ref scheme is exactly this special case of BME.
Phase 0 verifies this against the code. If confirmed: expose it as `coordination_mode="vref"` implemented **through** the BME machinery (quadratic-boundary objective plug-in), keeping the original implementation available for regression comparison. If not confirmed: document the delta precisely in the status file and keep both paths separate. Either way the chapter narrative gains: v_ref is the intuition, BME its formal generalisation.
### 3.8 Discrete hygiene protocol
Three ingredients; the first generalises an existing vertical mechanism horizontally.
1. **Switch notices (horizontal feedforward).** When area i commits a discrete move Δu_i_d ≠ 0 at step k, it publishes
   `SwitchNotice{area=i, step=k, dv_b^pred = H_{b,i}^d · Δu_i_d}`.
   Receivers use it exactly as the existing DSO feedforward correction uses MSR switch predictions: (a) correct the interpretation of the next boundary measurement so the neighbour's OFO does not misattribute the step to drift and counter-react; (b) mask/feed the correction to any online sensitivity estimator so it is not poisoned by the transient. Reuse the existing correction pattern; do not reinvent it.
2. **Discrete slotting.** Area i may **commit** discrete moves only when `k mod N_A == slot_i` (round robin; continuous moves every step). This deterministically prevents two areas from counter-switching on the same stale marginals in the same step. (**DECISION D5**: round robin vs token passing; slot length default 1 step.)
3. **ε-improvement acceptance.** After solving the MIQP, also solve the continuous-only QP with integers frozen at their current values. Commit the discrete part only if
   `Φ̂(MIQP) ≤ Φ̂(QP_frozen) − ε_switch − c_switchᵀ|Δu_i_d|`
   where Φ̂ is the local quadratic model prediction and c_switch a per-device switching cost (**DECISION D6**). Otherwise commit the continuous-only solution and hold the integers.
   **Log every decision** — predicted ΔΦ, realised ΔΦ (evaluated one step later from measurements), accepted/rejected, device class — into a switching ledger. This ledger is the empirical premise data for the finite-switching argument (§3.10).
Cold start and dropped messages: for the first d steps (no μ received yet) the controller runs explicitly in uncoordinated mode and logs this; it is a documented policy, not a silent default. After warm-up, a missing expected signal raises, unless drop simulation is enabled in config, in which case the explicit policy is hold-last-filtered-value (logged per occurrence).
### 3.9 Information and communication model
Honest accounting — this table goes verbatim into the status file and later the chapter:
| Quantity | Available to area i? | How |
|---|---|---|
| Own model, own measurements, own Φ_i | yes | local |
| v_b^meas at adjacent boundary buses | yes | boundary buses are jointly observable |
| μ_j from neighbours | yes, delayed d steps | CoordinationBus |
| SwitchNotice from neighbours | yes, delayed d steps | CoordinationBus |
| μ_i (self) | yes | local reduced Jacobian (§3.4) |
| H_{b,i} = ∂v_b/∂u_i | **concession** | depends on the global network response |
| Neighbour models, internal measurements, objectives | **no** | never exchanged |
The one informational concession is H_{b,i}: the response of boundary voltage to my own actuators depends on how the neighbouring network reacts. In simulation, take it from the global pandapower Jacobian with an **access-restriction wrapper**: area i's controller may only read rows(B) × cols(u_i) — enforce this in code (a provider object that raises on any other access) so the informational claim is verifiable, not rhetorical. In reality, H_{b,i} is exactly what the online sensitivity-estimation work (Kalman/RLS with directional forgetting) would identify from local input moves and boundary measurements; note this tie-in in the status file as the realism story, out of implementation scope here.
Communication: in-process publish/subscribe bus, explicit message dataclasses, configurable integer delay d and optional drop probability. No hidden global state; areas interact with the bus and the Plant only.
### 3.10 Theoretical characterisation targets (numerical, not proofs)
Consistent with the dissertation decision to replace formal proofs by empirical characterisation:
1. **Symmetrisation.** Assemble the coupled quadratic model: C_coop = Hᵀ Q_Φ H with the full cross-area blocks (Q_Φ = Hessian of Φ in output space at the nominal point, piecewise per §3.3), and M_coop = G_w^{−1/2} C_coop G_w^{−1/2}. Verify numerically: M_coop symmetric PSD ⇒ real spectrum ⇒ contraction condition α < 2/λ_max(M_coop) via the existing analysis ρ(I − α·G_w^{−1}C) < 1. Produce the comparison figure: eigenvalue cloud of the **non-cooperative** M (nonsymmetric off-diagonal blocks M_TSO,ij, complex eigenvalues, the existing oscillation mechanism) vs the real spectrum of M_coop. Cooperation symmetrises the game — this is the headline theory figure.
2. **Finite-switching premise.** With the ε-acceptance rule, Φ is bounded below on the compact feasible set and each accepted switch is predicted to decrease Φ by at least ε_switch; the number of switches is finite **if predictions are reliable**. The Monte Carlo campaign (§6) validates precisely that premise from the switching ledger: the empirical distribution of realised ΔΦ conditional on acceptance, and P(realised ΔΦ ≤ −ε_switch/2). The lemma pair (finite switching + between-switch contraction under the α condition) is then stated with the MC-validated premise made explicit.
3. **Delay honesty.** The symmetrisation statement holds for the synchronous ideal (d = 0, β = 1). Delay and filtering perturb it; characterise empirically via the delay sweep in §6 rather than claiming robustness.
---
## 4. Interfaces (target shapes, adapt to repository idioms in Phase 0)
Minimal signatures; fit naming to the existing codebase, keep the semantics.

```python
@dataclass(frozen=True)
class MarginalSignal:
    area_id: str
    step: int
    mu: np.ndarray            # length |B|, global boundary registry order, sparse-by-zeros
    v_b_meas: np.ndarray      # snapshot used to compute mu (for diagnostics)
@dataclass(frozen=True)
class SwitchNotice:
    area_id: str
    step: int
    dv_b_pred: np.ndarray     # length |B|
    devices: tuple[str, ...]  # device identifiers that moved
class BoundaryTopology:
    """Built once from the pandapower net + area partition.
    Provides: boundary bus registry (global order), per-pair sets B_ij,
    adjacency, ownership map (D1), and the separator assertion."""
class CommonObjective:
    """Per-area Φ_i and gradients; global Φ for tests/oracle.
    Guarantees partition invariant Σ_i Φ_i == Φ_global."""
class MarginalComputer:
    """Area-local: μ_j via internal reduced Jacobian with adjacent
    boundary buses as ports. Raises if separator assumption violated."""
class RestrictedSensitivityProvider:
    """Wraps the global H; area i may read only rows(B) × cols(u_i)
    (and its own internal rows). Any other access raises."""
class CoordinationBus:
    """In-process pub/sub with integer step delay d and optional drop
    probability. Deterministic under fixed seed."""
class DiscreteHygiene:
    """Slotting schedule, ε-acceptance check (MIQP vs frozen-integer QP),
    switching ledger writer, notice emission/consumption."""
```

Config addition (extend the existing config system, do not invent a parallel one):

```
coordination:
  mode: none | vref | bme
  delay_steps: 1
  drop_probability: 0.0
  beta_filter: 0.3
  epsilon_switch: <D6>
  switch_cost: {oltc: <D6>, shunt: <D6>}
  slotting: round_robin
  ownership: default_50_50
```

---
## 5. Build plan — phases, tasks, acceptance criteria
Commit at the end of each phase. Update `BME_STATUS.md` before and after each phase.
### Phase 0 — Repository reconnaissance (read-only; no code changes)
Tasks:
1. Locate and record in a **component-mapping table**: the multi-area simulation module (`multi_area_fes.py` or successor), the per-area TSO controller class and its per-step MIQP assembly, the sensitivity provider (where H comes from), the CAIR implementation, the hysteresis quantiser and dwell logic, the DSO feedforward correction, the implemented TSO–TSO v_ref scheme, the config system, the experiment runner for V1–V5, and the test layout.
2. **Audit A1 (gradient convention):** determine whether the current gradient assembly corresponds to Convention A or B of §3.5. Record the exact formula as implemented, with file/line references.
3. **Audit A2 (v_ref hypothesis):** check §3.7 against the v_ref code; record confirmed / delta.
4. **Audit A3 (boundary observability):** confirm boundary buses and tie-lines in the IEEE 39 3-area configuration; list them; check the separator property by inspection (formal check comes in Phase 1).
5. List open questions and any surprises.
Acceptance: `BME_STATUS.md` contains the mapping table, A1–A3 results with file/line references, and an explicit go/no-go note. **No source files modified.**
### Phase 1 — Boundary topology and sensitivities
Tasks: implement `BoundaryTopology` (registry, B_ij, ownership map per D1, separator assertion via graph check — hard error on violation); extend the sensitivity layer with `H_b_i(area)` (row/col selection of global H, discrete columns included) behind `RestrictedSensitivityProvider`; implement `MarginalComputer` (reduced internal Jacobian, ports = adjacent boundary buses).
Tests (all fail-fast, explicit tolerances):
- Separator check passes on the 3-area IEEE 39 case; a deliberately broken partition raises.
- Finite-difference validation of H_{b,i} columns (perturb each actuator incl. one tap step, compare Δv_b) on the 3-area case.
- Finite-difference validation of μ_j (perturb one adjacent boundary voltage as a port, compare dΦ_j) per area.
- Sparsity: μ_j entries at non-adjacent boundary buses are exactly zero.
- RestrictedSensitivityProvider raises on out-of-scope access.
### Phase 2 — Common objective
Tasks: implement `CommonObjective` with ownership map; per-area Φ_i, ∂Φ_i/∂v_b, internal gradients as needed by the audited convention; global Φ for tests and the oracle.
Tests: partition invariant Σ_i Φ_i == Φ_global at ≥ 20 randomised operating points; finite-difference check of ∂Φ_i/∂v_b; hinge behaviour at band edges (value and one-sided gradients).
### Phase 3 — Coordination bus and signals
Tasks: dataclasses, `CoordinationBus` with delay d and drop probability, receiver-side low-pass filter, cold-start policy (§3.8), determinism under fixed seed.
Tests: delay semantics (message published at k is visible at k + d, not earlier); cold start logs and runs uncoordinated for exactly d steps; missing-signal-after-warm-up raises when drops disabled; hold-last-value policy engages and logs when drops enabled.
### Phase 4 — Controller integration (the core)
Tasks: add `coordination_mode` to config and controller; implement gradient augmentation per the audited convention (A or B, §3.5); wire μ publication and consumption into the per-step sequence: measure → compute μ_i → publish → assemble g_i^bme with filtered delayed neighbour μ → solve MIQP → (Phase 5 gates) → apply.
Tests (regression + identity, the heart of the spec):
- `mode="none"` produces trajectories identical to the pre-BME baseline on an existing scenario (bitwise if solver-deterministic, else within solver tolerance — state which).
- Single-area identity: `mode="bme"` == `mode="none"` on a one-area network.
- **Distributed = centralised gradient** (test 2 of §3.5) with d = 0, β = 1, at ≥ 10 randomised points. Hard gate: do not proceed on failure.
- `mode="vref"` still reproduces the previously recorded v_ref results.
### Phase 5 — Discrete hygiene
Tasks: switch-notice emission and receiver-side correction (reuse the DSO feedforward pattern — Phase 0 mapping says where); estimator masking hook (no-op if no online estimator active, but the hook exists); slotting schedule; ε-acceptance with frozen-integer QP comparison; switching ledger (append-only, schema: step, area, device, predicted ΔΦ, realised ΔΦ filled at k+1, accepted flag, slot info).
Tests: constructed two-area scenario that counter-switches without slotting and does not with slotting (fixed seed, deterministic); ε-acceptance rejects a constructed marginal-benefit switch and accepts a constructed clear-benefit switch; ledger schema round-trips; notice correction changes the neighbour's innovation as predicted on a scripted switch.
### Phase 6 — Evaluation ladder and Monte Carlo campaign
Configs (one command each, reusing the existing experiment runner):
- (a) `none` — uncoordinated baseline (current non-cooperative behaviour),
- (b) `vref` — implemented reference scheme,
- (c) `bme` — this design,
- (d) `oracle` — centralised per-step OFO-MIQP: global Φ, all inputs in one solve, no communication constraints (upper bound; same solver and step logic).
Metrics module (computed uniformly for all rungs): Φ trajectory and settling; total switch count per device class; gap to oracle (Φ integral and terminal); per-TSO **normalised overcost** (Phulpin's fairness metric — answers "why would a sovereign TSO accept this" and shows whether the common objective creates net losers); boundary voltage band-violation time; oscillation indicator on boundary voltages (dominant AR pole or spectral peak).
Monte Carlo campaign: sweep load scenarios × delay d ∈ {0, 1, 2, 5} × sensitivity error (perturbed H) × β × ε_switch; fixed seeds; results to parquet + a generated summary markdown. Primary purpose: populate the switching ledger across conditions to validate the finite-switching premise (§3.10.2), and characterise delay sensitivity (§3.10.3).
Acceptance: each rung reproducible by one command; ladder comparison table and standard plots generated; MC results stored with seeds.
### Phase 7 — Analysis artefacts
Tasks: assemble C_coop, M_coop numerically; verify symmetry/PSD; compute spectra; produce the eigenvalue-cloud comparison figure (non-cooperative M vs M_coop) and the α-bound; ledger histograms (predicted vs realised ΔΦ, acceptance rates); `BME_RESULTS.md` summarising (i) the symmetrisation evidence, (ii) the finite-switching premise evidence, (iii) ladder + fairness outcomes, each with pointers to figures/data.
Acceptance: `BME_RESULTS.md` complete; every claim in it backed by a generated artefact; status file closed out with ✅ per phase.
---
## 6. Experiment design notes
- The ladder (a)–(d) mirrors the V1–V5 variant-ladder style of the CIGRE paper: each rung isolates one mechanism. Keep scenario definitions shared across rungs — only `coordination.mode` differs.
- The headline claims the experiments must be able to support or refute: (1) BME removes/damps the oscillatory interaction present in (a) (spectral indicator + eigenvalue figure); (2) BME closes most of the gap between (a) and (d) (quantify %); (3) no TSO is a systematic net loser under Φ (fairness metric); (4) discrete hygiene keeps switching counts at or below (a) levels while improving Φ; (5) performance degrades gracefully with d and sensitivity error.
- Log everything needed to recompute metrics offline; never compute-and-discard inside the loop.
---
## 7. DECISION points — surface to Manuel before implementing
| ID | Decision | Default proposal | Notes |
|---|---|---|---|
| D1 | Tie-line/boundary ownership convention | losses 50/50 per tie-line; each boundary bus's band penalty to exactly one adjacent area | only affects the partition of Φ, not its total |
| D2 | Common-objective weights w_loss, w_band, soft-band edges | w_loss = 1; w_band large enough that band violations dominate near edges; edges at ±3% around nominal | propose a small calibration sweep in Phase 6 |
| D3 | Filter constant β | 0.3 | sweep in MC anyway |
| D4 | Communication delay d | 1 step | sweep {0,1,2,5} in MC |
| D5 | Slotting scheme | round robin, slot length 1 step | token passing as alternative if round robin shows idle-slot cost |
| D6 | ε_switch and per-device c_switch | calibrate relative to the empirical per-step ΔΦ of the continuous loop (e.g. ε_switch ≈ 5× median per-step improvement); c_switch from device-wear reasoning | Phase 6 sweep; document rationale |
| D7 | Boundary quantity | voltage magnitudes only | angles noted as limitation in the chapter |
| D8 | Oracle variant | centralised per-step OFO-MIQP | optional additional static MIQP-OPF bound if cheap |
---
## 8. Conventions and guardrails (binding)
- **Fail-Fast everywhere.** `rep1()`-style assertions / explicit raises; no silent defaults, no `try/except: pass`, no fallback sensitivities, no auto-generated dummy data. A missing precondition is an error, not a warning.
- **British English** in all comments, docstrings, logs, and generated markdown.
- Every new module starts with a header comment mapping its code symbols to §3 symbols (e.g. `mu ↔ μ_j (§3.4)`, `H_b_i ↔ H_{b,i} = ∂v_b/∂u_i (§3.5)`).
- Do not refactor unrelated code; do not touch V1–V5 configs or vertical-cascade modules except at the explicitly named integration points.
- Notation in any emitted LaTeX: sentence-per-line, `% DRAFT` markers, `\gls{}`, `\MAT{}`, `\VEC{}`, superscript = value qualifier, subscript = quantity/index, `% TODO[refs]` placeholders.
- Tests accompany each phase; a phase without green tests is 🚧, not ✅.
- Update `BME_STATUS.md` at phase boundaries; record every DECISION outcome there with date.
- Determinism: fixed seeds for anything stochastic; MC results must be reproducible.
---
## 9. Positioning references (for the status file and later the chapter)
- Phulpin, Begovic, Petit, Ernst (2009), *A fair method for centralized optimization of multi-TSO power systems*, IJEPES — closest prior art; explicitly rejects inter-TSO objective weighting as arbitrary; centralised, continuous, model-based. BME advances it on all four axes (distributed, feedback-based, mixed-integer, per-step).
- Colombino, Dall'Anese, Bernstein — *Online Optimization as a Feedback Controller* (arXiv:1805.09877) — foundational OFO stability/tracking.
- Distributed coupled-subsystem OFO: arXiv:2302.09241 (reactive sharing under voltage limits, singular-perturbation stability); arXiv:1912.07926 (distributed primal–dual for AC networks).
- ADMM at TSO scale: arXiv:1912.03942 (European grid, border-only exchange) — supports the "coordinate over boundary quantities, keep everything else local" design principle; BME differs by staying single-solve-per-step (no inner iterations) and by pricing integers inside the local MIQP.
- Discrete/continuous separation literature: double-time-scale MPC+ADMM (ScienceDirect S0142061522006615); discrete-device Volt/VAR (arXiv:2601.22080) — the relax–round–resolve contrast for §3.6.
---
*End of spec. Begin with Phase 0 and report the component-mapping table and audits A1–A3 before writing any code.*
