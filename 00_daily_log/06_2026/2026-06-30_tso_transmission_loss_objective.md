# TSO transmission-loss objective (form B, current-magnitude)

**Date:** 2026-06-30
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Add an optional active-transmission-loss term to the TSO OFO/MIQP
objective, reusing the existing cached sensitivities. Leave a hook for a future
PMU (voltage-phasor) loss form.

## What was added

An optional objective term

    f_loss = g_loss · P_loss ,   P_loss = Σ_ℓ c_ℓ · |I_ℓ|²        (form B)

summed over the zone's **monitored current lines** (`current_line_indices`),
with default per-line coefficient `c_ℓ = 3·R_ℓ` [MW per kA²]
(`R_ℓ = r_ohm_per_km·length_km/parallel`, three-phase I²R). Default `g_loss=0`
⇒ term entirely absent (legacy behaviour preserved).

### Key method / structure of change

The loss is a function of the **current outputs**, which already have rows in
the cached sensitivity matrix `H` (`∂I_ℓ/∂u`, in kA, built from the existing
Jacobian primitives `compute_dI_d*`). So the loss needs **no new sensitivity
primitive** — it enters purely through the (previously zero) `I` block of the
output-space gradient:

    ∂(g_loss·P_loss)/∂I_ℓ = 2·g_loss·c_ℓ·|I_ℓ|        (anchored on measured |I_ℓ|)

and is projected to control space via the cached current rows:

    Δgrad_f = (2·g_loss·c·|I_meas|) · H[I-rows]

Because the **same** `_loss_output_grad_i(measurement)` vector is used in both
`_compute_output_gradient` (the `I` block) and `_compute_objective_gradient`
(Component 6, projected through `H[I-rows]`), the pinned invariant
`grad_f == ∇_y f · H` (`tests/test_tso_output_gradient.py`) holds by
construction. A welcome side effect: the switched-shunt integrator, which dots
`∇_y f` with each bank's boundary sensitivity column, automatically becomes
loss-aware and follows the same objective.

The measured `|I_ℓ|` is the OFO linearisation anchor, so the loss gradient is
re-anchored on the true operating point every step — consistent with the
"controller never sees the plant, only cached sensitivities + measurements"
principle.

### Files changed

- `controller/tso_controller.py`
  - `TSOControllerConfig`: new `g_loss` (default 0.0), `loss_line_coeff_mw_per_ka2`
    (optional per-line override), `loss_use_phasor` (PMU hook, default False);
    `__post_init__` validation (non-negative weight, length/sign of override,
    `g_loss!=0` requires ≥1 monitored current line).
  - `TSOController`: `_loss_coeff_cache`; new `_loss_line_coeffs()` (default
    `3·R_ℓ` from `sensitivities.net` — the same net that builds `∂I/∂u`),
    `_loss_output_grad_i()` (form B), `_loss_output_grad_i_phasor()` (guarded
    stub). Wired `grad_i` in `_compute_output_gradient`; added Component 6 to
    `_compute_objective_gradient`.
- `core/measurement.py`: new optional `voltage_angles_deg` field (PMU channel),
  populated from `net.res_bus.va_degree` in `measure_tso`, `measure_zone_tso`,
  `measure_central`. Default empty; ignored by the magnitude-only path.
- `configs/multi_tso_config.py`: `tso_g_loss`, `tso_loss_use_phasor` pass-throughs.
- `experiments/runners/multi_tso_dso.py`: route both into `TSOControllerConfig`.
- `tests/test_tso_loss_objective.py`: new — invariant under loss, default `3R`
  coefficient, current-reducing/non-zero gradient, override-zeroing a line,
  phasor stub raises, empty-current-line config guard.

## Assumptions / model facts

- Actuators are reactive only (AVR V_set, OLTC, MSC/MSR, TS-DER Q); active
  dispatch is fixed. Losses are controllable through the voltage-magnitude /
  current pathway (higher V ⇒ lower I for the same P,Q ⇒ lower I²R).
- `|I_ℓ|` = from-side current magnitude (`i_from_ka`), matching the H current
  rows. Using `i_from²` (not the from/to average) is a documented approximation,
  acceptable since the controller works off a cached/linearised model anyway.
- Loss sum spans exactly `current_line_indices`. To count more in-zone branches,
  add them there — their `∂I/∂u` rows then carry the gradient. (This couples
  loss-line selection with thermal monitoring; a per-line `0.0` override lets a
  line be monitored for limits but excluded from the loss sum.)
- Implemented as a **gradient-only** term (projected-gradient OFO), exactly like
  voltage tracking and the reserve terms — no Hessian added to the QP.

## Risks / unresolved points

1. **Voltage-bound interaction.** Loss minimisation raises the voltage profile
   (less current for fixed P) and will ride the upper `v_max_pu` limit; the hard
   voltage band must stay enforced. This is the main coupling to watch in tuning.
2. **Weight scale.** `g_loss` competes with `g_v` (primary) and `g_w` movement
   penalties; loss is a small, slow, global cost. Treat as tertiary — start
   small. Loss is in MW, so the gradient magnitude is `2·g_loss·(3R)·|I|`; size
   `g_loss` against the voltage term's `2·g_v·ΔV` after one trial step.
3. **Coverage.** Only monitored lines are counted; the reported `P_loss` proxy is
   a subset of true zone losses unless `current_line_indices` is expanded. Not a
   correctness bug, but the objective is "loss over monitored lines", not "total
   zone loss". Document in any results.
4. **PMU path is a stub.** `loss_use_phasor=True` raises `NotImplementedError`.
   The exact form `P_loss = Σ g_ℓ(V_i²+V_j²−2V_iV_j cosθ_ij)` has V-block as well
   as I-block gradient components, so wiring it will extend
   `_compute_output_gradient` (and its consistency test) beyond the current
   rows. Angles are already carried on `Measurement.voltage_angles_deg`.

## Comparison experiment — `009_TSO_LOSS_TIE_SWEEP.py`

Added a simple 2-factor sweep to study the loss term and its interaction with
the horizontal tie coordinator:

- **Factor 1 — loss weight** `tso_g_loss ∈ {0, 1e3, 1e4, 1e5}` (0 = baseline).
- **Factor 2 — tie coordinator** `enable_tie_coordination ∈ {off, on}`.

Cross product → 8 scenarios; everything else fixed by `make_base_config`
(cascade OFO both layers, IEEE39 `wind_replace`, clean 60-min profile, no
contingencies so loss deltas are pure reactive-redispatch effects). Primary
scorer is the **plant** loss `MultiTSOIterationRecord.total_losses_mw`
(whole-network PF loss), independent of the controller's monitored-line proxy —
so it fairly tests whether minimising the proxy lowers real losses. Secondary:
voltage RMSE / envelope (loss-vs-voltage trade-off) and mean `Σ|Q_tie|`
(coordinator effect).

Outputs under `results/009_loss_tie_sweep/`: `summary.csv`,
`losses_vs_gloss`, `losses_timeseries`, `tie_q_vs_gloss` (png+pdf). CLI:
`--smoke` (2-step wiring check), `--only <names>`, `--replot`.

Wiring validated with `--smoke` (the term is active and changes
`total_losses_mw`; CSV + figures generate NaN-safely for un-run cells). The
full 8-scenario × 60-min sweep is left for the user to run (~tens of minutes
wall-clock).

**Open point — weight scale.** The `tso_g_loss` magnitude that "bites" is
scenario-dependent (it competes with `g_v=3e5` and the `g_w` move penalties
through the relative sizes of `∂I/∂u` vs `∂V/∂u`). The chosen sweep spans 3
decades; if the full run shows a negligible or over-strong effect, shift
`LOSS_WEIGHTS` at the top of the experiment. A principled first guess: size
`2·g_loss·(3R)·|I|·‖∂I/∂u‖` to be a small fraction of the voltage gradient
`2·g_v·ΔV·‖∂V/∂u‖` at a representative step.

## Next note to update in Obsidian

`[[todo]]` — mark "TSO loss objective (form B)" done; add follow-ups:
(a) tune `tso_g_loss` vs voltage tracking on `002_M_TSO_M_DSO_COMPARE`;
(b) decide whether to widen `current_line_indices` to all in-zone branches for a
true zone-loss objective; (c) implement the PMU phasor loss form when phasor
measurements are introduced.
