# 2026-06-23 — Tier-2 curvature-based g_w preconditioning (auto-κ)

**Author:** Manuel Schwenke (with Claude Code)
**Scope:** Generalise the manual V5 curvature probe
(`_dump_central_curvature`) into a reusable, automatic rule that derives the
proximal weights `g_w` of the *continuous* actuator classes from each
controller's cached sensitivities — no Bayesian optimisation, no closed-loop
simulation. Behind a config flag; the BO/config path is untouched when off.

## Rationale (Tier-2 of the weight-tuning taxonomy)

`g_w` is **not** a control preference — for fixed objective ratios it is the
OFO loop gain. One unconstrained tick is
`σ* = −G_w⁻¹ H_Vᵀ diag(g_v)(V−V*)`, so the voltage error obeys
`e_{k+1} = (I − M) e_k` with **`M = H_V G_w⁻¹ H_Vᵀ diag(g_v)`** (stable iff
`eig(M) ⊂ (0,2)`, well-damped at `λ_max(M) ≲ 1`). Writing `A = D_v^{1/2} H_V`,
`M_sym = Σ_i (1/g_w_i) a_i a_iᵀ`: column `i` contributes `‖a_i‖²/g_w_i`. Hence:

- **Shape (conditioning):** `g_w_i ∝ ‖a_i‖²` equalises the per-actuator
  contributions — exactly the diagonal scaling matrix `S` of Zagorowska et al.
  (IFAC WC 2026, Eq. 16) in this project's `g_w_i ∝ 1/S_i` convention.
- **Gain (scale):** since `M ∝ G_w⁻¹`, one scalar `κ` on the whole block places
  `λ_max(M)` at any target (the same cooling hand-applied to V5, `κ=1.25`).

So the closed-form rule replaces a BO sub-search that is slow, opaque, and
gameable (cf. the `g_w_pcc`→freeze-PCC exploit in `tuning/metrics.py`).

## What was added

**New module `controller/gw_precondition.py`** (pure, controller-agnostic):
- `curvature_spectrum(H_v, g_v, g_w) → CurvatureSpectrum` — eigenvalues /
  `λ_max` / `λ_min⁺` / `cond⁺` of the symmetric form
  `D_v^{1/2} H_V G_w⁻¹ H_Vᵀ D_v^{1/2}`. **Shared** with the V5 probe (DRY).
- `precondition_g_w(...) → PreconditionResult` — (i) column-norm precondition
  the listed continuous classes (`'class'` = one shared `g_w` per class,
  comparable to a BO `g_w_<class>`; `'column'` = full Zagorowska-`S` diagonal),
  (ii) solve one `κ` (monotone log-bisection, clamped `[1e-8, 1e12]`) so
  `λ_max(M) = lambda_target`. Floors near-zero columns; if the **fixed**
  (integer/OLTC) curvature alone already exceeds the target it **declines to
  act** (no-op + `target_feasible=False`) rather than freezing the DERs or
  worsening `λ_max` — because that is a switching-cadence problem (Tier-2′),
  not a `g_w` problem.

**Controllers** — `voltage_curvature_inputs() → (H_v, g_v_vec)` returning the
voltage block of the *expanded* H in each controller's own output ordering:
- `tso_controller.py`: rows `[:n_v]` (V first), `g_v = config.g_v`.
- `dso_controller.py`: rows `[n_interfaces : n_interfaces+n_v]` (V follows the
  interface-Q block); returns `None` when no voltage schedule is active (a
  Q-dominant DSO has no voltage curvature to precondition — interface-Q
  curvature is the documented next extension).
- `central_controller.py`: rows `[:n_v]`, `g_v = g_v_per_bus`.
- `base_controller.py`: default `voltage_curvature_inputs()→None`, plus
  `apply_preconditioned_g_w(vec)` (installs the final per-variable vector
  verbatim — DER-incidence weighting already baked into the H columns — and
  rebuilds the online `GwAdapter` so Tier-3 warm-starts from the Tier-2 gain).

**Config** (`configs/multi_tso_config.py`): `precondition_g_w=False`,
`precondition_lambda_target=0.9`, `precondition_granularity='class'`,
`precondition_floor_frac=1e-6`, `precondition_exclude_classes=('gen',)`.

**Runner** (`experiments/runners/multi_tso_dso.py`): refactored
`_dump_central_curvature` onto `curvature_spectrum`; added
`_apply_gw_preconditioning(...)` and a hook after controller `initialise(...)` /
numerical-H pin (so it preconditions the H the MIQP actually uses). Continuous
classes are auto-detected (indices ∉ `_integer_indices`) minus
`precondition_exclude_classes`.

## Key design decisions

1. **Integers excluded by construction.** OLTC/shunt `g_w` is left at config;
   their tuning primitive is switching frequency (deadband/cadence), not
   curvature — consistent with the new `ShuntIntegrator`.
2. **`gen` (AVR V-setpoint) excluded by default.** It is a *direct* strong
   voltage actuator (column energy ~10^10× a DER's) the user already pins out
   of stability tuning (`FIXED_OVERRIDES`); folding it into the shared `κ`
   makes it dominate. Configurable via `precondition_exclude_classes`.
3. **Voltage-only curvature** (matches the validated V5 probe). Extending to
   the full output-weighted `M` (add `g_q`, `g_res`, and the DSO interface-Q
   block) is a drop-in — the rank-1 algebra is unchanged.

## Validation

- **Unit** `tests/test_gw_precondition.py` (17): `M ∝ 1/g_w` scaling; `λ_max`
  hits target for `class`/`column` × targets {0.5, 0.9, 1.2}; integers
  untouched; column-granularity equalises `‖a_i‖²/g_w_i = 1/κ`; near-zero
  column floored; integer-dominated → no-op + flag; scale-covariance in
  `H`, `g_v`; input validation. **17 passed.**
- **Controller suite** (`test_controller`, `test_g_w_adapter`,
  `test_tso_output_gradient`, `test_tso_saturation`, `test_oltc_cooldown`):
  **122 passed, 1 skipped** — additive changes, no regression.
- **End-to-end smoke** (`002_M_TSO_M_DSO_COMPARE`, 3-min horizon,
  `precondition_g_w=True`): completes; flag off = byte-identical BO path.
  Feasible zones (TSO-z2 + all 4 DSOs) driven to `λ_max=0.900` with sane
  `g_w` (der/pcc≈5.9, dso_der≈0.24). **Finding:** TSO-z1, z3 are
  **INTEGER-DOMINATED** (`λ_floor` = 2.2 / 1.4 ≥ target): their OLTC voltage
  curvature alone exceeds the stability target at the current
  `g_w_tso_oltc` — the preconditioner correctly declines and flags it. The BO
  would have silently absorbed this; here it is surfaced as a diagnostic.

## Open / next

1. **Integer-dominated z1/z3** is a real signal: either `g_w_tso_oltc` is too
   small (OLTC modelled as a too-aggressive continuous iterator in `M`) or the
   OLTC genuinely needs cadence/deadband treatment. **Decide:** exclude
   integer columns from `M` entirely (they "settle finitely", per
   `analysis/stability_analysis.py`) vs. raise `g_w_tso_oltc`. Flagged for
   discussion — not changed unilaterally.
2. **Full output-weighted M** (g_q / g_res / DSO interface-Q) — the documented
   extension; needed before preconditioning Q-dominant DSOs.
3. **A/B vs BO** on `rms_v_ts_pu` (the variant-neutral metric) — the intended
   payoff: comparable voltage tracking at a fraction of the tuning cost.

## Note (pre-existing, NOT introduced here)

`tests/tuning/` has **8 failures** unrelated to this change: the committed
`FIXED_OVERRIDES` (`tuning/parameters.py`) lists `g_w_dso_der_vref`, which is
absent from `MultiTSOConfig` (both HEAD and working tree), so
`apply_to_config`/`dataclasses.replace` raise `TypeError`. This means the
**BO `apply_to_config` path is currently broken** independently of Tier-2 —
worth fixing (add the field or drop it from `FIXED_OVERRIDES`).

---

## Follow-up (2026-06-23): cap-only rule + A/B sweep harness

**Trigger.** A live run with `precondition_lambda_target=0.9` made the cascade
*oscillate*. Diagnosis (established): the original BO/config loops were already
well-damped (`λ_max` 0.30–0.86); forcing `λ_max = 0.9` *raised* every loop's
gain — especially the Q-dominant DSOs (0.30→0.90, ~3× hotter, sized off the
wrong voltage-only `M`) — stripping the cascade/model-error margin. `λ_max=0.9`
is a *ceiling* for stability (`eig(M)⊂(0,2)`), not a setpoint to sit at; under
cached-`H` model error and TSO↔DSO coupling you want margin well below 1.

**Change — cap-only semantics.** `precondition_g_w` may now only *reduce* a
loop's `λ_max`, never raise it. `PreconditionResult.target_feasible` →
`status ∈ {reduced, within_margin, integer_dominated, no_class}` + `applied`:

| `λ_max_before` vs target | action |
|---|---|
| `> target` and `> λ_floor` | **reduced**: reshape + scale DOWN to `λ_max=target` |
| `≤ target` | **within_margin**: no-op (acting could only make it hotter) |
| `λ_floor ≥ target` | **integer_dominated**: no-op + flag (OLTC binds — Tier-2′) |

So `target` reads as *"guarantee every loop is at least this damped."* With
`target=0.9` on the current case every loop is within-margin ⇒ **no-op ⇒
byte-identical to BO** (verified: `rms_v_ts_pu` identical). Files touched:
`controller/gw_precondition.py` (logic + result fields), runner print
(`REDUCED` / `within margin` / `INTEGER-DOMINATED` tags), tests rewritten.

**New harness `experiments/diag_precond_sweep.py`.** Sweeps
`precondition_lambda_target` and prints the controller-agnostic KPI
`rms_v_ts_pu` (via `cigre_summary_table`) + `n_sw` per target against the
`precondition_g_w=False` baseline — turning "guess a target" into "read the
speed↔margin knee." CLI: `--module --targets --horizon-min --scenario
--granularity --verbose`.

**First read (002, 5-min, no contingency):**

| variant | rms_v_ts_pu | Δ vs BO | note |
|---|---|---|---|
| BO_baseline | 0.014884 | +0.0% | |
| precond_0.3 | 0.015912 | +6.9% | hot TSO loops pulled to 0.3 → safer, slightly slower |
| precond_0.9 | 0.014884 | +0.0% | all loops ≤0.9 → within-margin → identical to BO |

The real comparison wants a longer horizon with contingencies/disturbance —
where the conservative margin (or BO's over-driven zones) actually bites.

**Tests:** `tests/test_gw_precondition.py` rewritten for cap-only (17 passed:
reduces-hot-loop, within-margin no-op, integer-dominated declines,
column-equalisation, floor, scale-covariance). Controller suite still green
(107 passed, 1 skipped).

**Open:** still need (a) the `g_w_dso_der_vref` BO fix (above), (b) full
output-weighted `M` so Q-dominant DSOs precondition off `g_q` not voltage, and
(c) a long-horizon A/B to decide whether Tier-2 matches BO with one knob.

---

## Follow-up (2026-06-23): Tier-1 objective weights — normalisation + priority + sweep

Picking up the *weight*-tuning methodology (NOT sensitivities). With `g_w`
handled by the preconditioner (Tier-2), the objective weights are now
**decoupled from stability** — changing a ratio re-derives `g_w` to hold
`λ_max`, so objective tuning can no longer destabilise the loop. Settled
Steps A/B and built the sweep tool.

**Step A — normalisation.** Objective written dimensionless,
`f = Σ_i π_i (e_i/σ_i)²`, so the controller weight is *derived*:
`g_i = π_i / σ_i²`. Tolerances `σ` (fixed, from `metrics.py`/physics):
`σ_v,ts=5 mpu`, `σ_v,ds=10 mpu`, `σ_q,pcc=5 Mvar`, `σ_q,tie=20 Mvar`,
`σ_res=1` (already normalised). The raw `g` look extreme only because of the
pu-vs-Mvar unit conversion buried in `1/σ²`.

**Step B — priorities `π` (the design choice).** TSO: voltage 100 (primary),
interface-Q 10, tie-Q 10, reserve 1 (lowest). DSO: interface-Q 100 (primary —
tracks the TSO setpoint), voltage 10 (soft; feasibility lives in `g_z`).
Derived raw: `g_v≈4e6`, `tso_g_q_pcc≈0.4`, `tso_g_q_tie≈0.025`,
`tso_g_res≈1`; DSO `g_q≈4`, `dso_g_v≈1e5`. **Finding:** the current
`tso_g_q_tie=10` (with `g_v=5e4`) implies tie-Q is weighted *far above*
voltage in priority space — likely unintended; normalisation surfaces it
(caveat: objective-value ≠ step dominance — the sensitivities also scale the
gradient, hence we verify with the sweep, not the paper priorities alone).

**New harness `experiments/diag_objective_ratio_sweep.py`.** Re-parameterises
the *config* (not the controllers) on `(π, σ)`; sweeps the two TSO ratios
`π_qpcc/π_v`, `π_res/π_v` with the **preconditioner ON** (stability held);
reports `cigre_summary_table` KPIs and marks the **Pareto front** of
`rms_v_ts_pu` (↓) vs `m_bar_mvar` (↑, reserve). `SIGMA`/`PI_BASE` are
module-level and editable. CLI: `--pi-qpcc --pi-res --target --horizon-min
--scenario --module`.

**Smoke (002, 4-min, 2×2):** runs; Pareto fires (zero-`π_qpcc` dominates —
interface-Q tracking slightly costs voltage with no reserve gain). Reserve
shows no effect on a static short run (generators unstressed) → the real read
needs a **long horizon with contingencies**; that is the intended Step-C run.

**Workflow now:** set `(π, σ)` → preconditioner derives `g_w` (stability) →
sweep `π` for the performance trade-off → pick the knee. One interpretable
priority vector replaces the 9-D BO weight search.
