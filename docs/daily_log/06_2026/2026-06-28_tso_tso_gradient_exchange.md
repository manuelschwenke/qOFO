# 2026-06-28 — TSO–TSO coordination: gradient-exchange redesign

**Author:** Manuel Schwenke (with Claude Code)
**Scope:** Replace the two-loop ΔV_ref coordinator's *outer law* (relaxation toward
the realised difference + optional reserve-economic anchor `tie_econ_gamma`) with a
**combined boundary-gradient descent**. Each zone now shares the marginal of its
**full OFO objective** w.r.t. its boundary voltage; the coordinator descends the
*joint* gradient with a per-zone worsening safeguard. This subsumes the reserve
heuristic (reserve is now intrinsic to the gradient via `g_res`) and resolves the
"what units is the exchange in?" question (objective units throughout).

## Mechanism

Per tie `e = (i, j)`, with boundary-V sensitivity row `h_b = ∂V_b/∂u`:

```
γ_i = (∇J_i · h_b) / (h_b · h_b)                 # marginal of FULL objective, not just tracking
G   = κ·γ_i − (1−κ)·γ_j                           # = ∂(J_i + J_j)/∂ΔV_ref
Δ   = −grad_alpha·G − anchor·DB_Δ(ΔV_ref)         # joint descent + weak subsidiarity
safeguard: dJ_i ≈ γ_i·κ·Δ ,  dJ_j ≈ −γ_j·(1−κ)·Δ
           if max(dJ_i, dJ_j) > grad_eps:  Δ ← Δ · grad_eps / max(dJ_i, dJ_j)
ΔV_ref ← Π_[±dvref_max]{ ΔV_ref + Δ }
```

`∇J_i` is the **iterating** control-space gradient (`_compute_objective_gradient`),
so — unlike the converged setpoint marginal — the envelope theorem does *not* zero
the reserve term: voltage tracking + reserve + effort all contribute. The inner
loop is unchanged (each zone tracks `V_ref = V_anchor ± split·ΔV_ref` via its
primary objective; an unreachable setpoint is just bounded tracking error).

Voltage-only relaxation is the special case `γ_i = 2·g_v·err_i` (`g_res = 0`).

## g_v-agnostic knobs (key tuning fix)

Both step and cap scale with the objective level so the user-facing knobs are
`g_v`-independent (runner-side):

```
grad_alpha = tie_grad_step / (2·g_v)     # tie_grad_step = Newton-step fraction (0.5)
grad_eps   = tie_grad_eps · g_v          # tie_grad_eps = worsening cap / unit-error objective (1e-3)
```

**Why this matters:** the CIGRE config runs `g_v = 1e7`, so boundary gradients are
`γ ~ 1e6` and objectives `~1e3`. With the first-cut absolute `grad_eps = 10` the
safeguard capped per-zone worsening three orders below the objective ⇒ ΔV_ref
frozen at ~1e-4 (silently inert — the same failure mode as the old price design).
Scaling `grad_eps` by `g_v` restores motion: in the divergent smoke ΔV_ref now
reaches ~0.05–0.065 (bounded by `dvref_max`), PF stays feasible.

## What changed

- `controller/tie_coordinator.py`: rewritten. `TieCoordinatorConfig` now
  `{grad_alpha, grad_eps, anchor, deadband_v_pu, kappa, dvref_max}` (removed
  `relax`, `econ_gamma`). `update(gradients={tie_id:(γ_i,γ_j)})` does the combined
  descent + safeguard. `state()` → `{dvref, grad_i, grad_j, grad_combined}`.
- `controller/tso_controller.py`: **+`report_boundary_gradient(meas, bus) → γ`**
  (`γ = ∇J·h_b/‖h_b‖²`, voltage block = leading H rows). The three
  normalized-utilisation reserve methods (`_reserve_num_den`,
  `report_reserve_scarcity`, `report_reserve_capability`) are now dead — left in
  place, flagged for removal.
- `experiments/runners/multi_tso_dso.py`: round collects `γ_i, γ_j` per tie via
  `report_boundary_gradient`, calls `update(gradients)`; reserve μ/headroom kept
  as **diagnostic only** (recorded, not fed to `update`). Records
  `tie_grad_i/j/combined`, computes `tie_dv_realized = V_i − V_j` in the runner.
  Coordinator built with the g_v-scaled `grad_alpha`/`grad_eps` above.
- `configs/multi_tso_config.py`: `tie_grad_step=0.5`, `tie_grad_eps=1e-3` (replace
  `tie_relax`); removed `tie_econ_gamma`; `tie_reserve_headroom_scale_mvar` kept,
  re-docstringed as a diagnostic.
- `experiments/helpers/records.py`: `tie_grad_i/j/combined` replace
  `tie_marginal`/`tie_econ_target`.
- `visualisation/plot_tie_coordination.py`: 4th tile now plots the combined
  gradient `G`; dropped the econ-target dotted overlay.
- `experiments/{000,005}`: `tie_relax=` → `tie_grad_step=` (constructor kwarg).
- `experiments/007_TIE_COORDINATION.py`: `--reserve` mode collapsed from
  OFF/COORD-sub/COORD-econ to **OFF/COORD** (sub≡econ now that reserve is in γ);
  the econ-target panel → combined-gradient `G` panel; μ scarcity/headroom kept as
  the diagnostic readout.
- `tests/test_tie_coordinator.py`: rewritten for the gradient API (18 tests:
  descent direction, per-zone safeguard cap, anchor, clip, validation, messages).

## Verification

- `test_tie_coordinator.py` 18 ✓; `test_tie_coordination_hooks.py` 8 ✓;
  `test_tso_output_gradient.py` (gradient invariant) ✓ — 26 total green.
- End-to-end runner smoke (divergent 1.05/1.03/1.02, 8 min, tie coord ON):
  finite γ/G/ΔV_ref on all 5 ties; ΔV_ref active (~0.05–0.065) and bounded; PF
  feasible throughout.

## Risks / open

- **Tuning not validated.** `tie_grad_eps=1e-3` makes the coordinator active but
  may be too hot (several ties near `dvref_max` in the short smoke). Needs the
  proper divergent + uniform validation sweep (à la the earlier two-loop runs:
  expect benefit in divergent, neutrality in uniform) before trusting defaults.
- **Absolute vs relative safeguard.** The cap bounds *absolute* per-zone worsening;
  it still throttles strongly-lopsided-but-net-beneficial trades. A *relative* cap
  (worsening as a fraction of the neighbour's improvement) may be more faithful to
  "help only if cheap for me" — a possible follow-up (design change, discuss first).
- **Losses deferred.** The framework admits a loss term once `∂P_loss/∂V` is
  available, which needs PMU angles / a full observable state estimate. Out of
  scope here by decision (voltage + reserve first).
- Dead reserve methods (`_reserve_num_den` et al.) left in `tso_controller.py`;
  remove in a dedicated cleanup once confirmed unwanted.
