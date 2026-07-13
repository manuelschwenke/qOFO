# 2026-06-22 — Integrator-based MSC/MSR switched-shunt dispatch

**Author:** Manuel Schwenke (with Claude Code)
**Scope:** Add a separate integrating dispatcher for TSO-owned mechanically
switched shunt banks (MSC + MSR) at the DSO tertiary windings, OUTSIDE the OLTC
MIQP, with a synchronised DSO interface-setpoint feedforward and an SMW-only
sensitivity refresh (no power flow on a switch).

---

## 1. Rationale (recorded per plan)

A per-instant MIQP engages a large bulk device only when the full step improves
the snapshot objective; with a ~50 Mvar step and the switching penalty needed
for stability, that rarely holds, so the bank is engaged very seldom or
chatters. An **integrating continuous-relaxation state** instead accumulates the
time-integral of the reactive "pressure" projected onto the bank's boundary
sensitivity and commits a physical step on **sustained** need — the correct
behaviour for a slow bulk device. OLTCs (small steps) stay in the MIQP; shunts
(large steps) use the integrator.

## 2. How shunt control worked BEFORE vs AFTER

| Aspect | Before (legacy) | After (this change) |
|---|---|---|
| Device | One **bipolar** ±1 bank/DSO, 50 Mvar | N-step **MSC** (capacitor) + **MSR** (reactor) banks/DSO |
| TSO dispatch | Integer `s_shunt` **inside the MIQP** | Separate `ShuntIntegrator`, **outside the MIQP** |
| Selection | Per-snapshot MIQP + `g_w` penalty + iteration cooldown | Anti-windup integrator + hysteresis quantiser + dwell + daily budget + HV feasibility guard |
| DSO view | Disturbance only (`shunt_bus_indices=[]`) | Unchanged: disturbance only |
| DSO interface FF | Implicit/joint via MIQP `Q_PCC` (off by default) | Explicit persistent per-interface offset `Q_itf_set += dQ_itf_sh`, atomic with the toggle |
| Mode switch | `install_tso_tertiary_shunts` (bool) | `shunt_dispatch ∈ {'off','miqp','integrator'}` — legacy kept behind the flag |

Backward compatibility: a legacy config that sets only
`install_tso_tertiary_shunts=True` (leaving `shunt_dispatch='off'`) is resolved
to `'miqp'` so existing experiments are unchanged.

## 3. Sensitivity refresh on a switch — SMW only, NO power flow (user constraint)

On a commit the cached sensitivities are updated **without any `pp.runpp`**. The
changed tertiary susceptance hits one diagonal of the bus admittance, so
`JacobianSensitivities.apply_shunt_step_change_smw` applies a rank-1
Sherman–Morrison correction to `J_inv` and `dV_dQ_reduced` in place (now
**index-aware** so an MSC and an MSR sharing a tertiary are disambiguated). The
H rebuild then refreshes the 3-winding interface sensitivities through that
updated Jacobian — both the DER column (`∂Q_HV-3W/∂Q_DER` via `dV_dQ_reduced`)
and the OLTC column (`∂Q_HV-3W/∂s_OLTC` via `J_inv`). The cached operating point
`(V, θ)` and the operating-point branch partials (`∂Q_HV/∂V_hv`, `∂g/∂τ`, the
`V²` scaling) are **held fixed** by design — controllers never see the plant;
the susceptance change is the dominant first-order term and the rank-1 update
captures it exactly. The TSO, DSO, and plant nets are independent deep copies,
so the plant toggle and the two SMW reads do not interfere.

## 4. Files changed

- **add** `controller/shunt_integrator.py` — `ShuntBankConfig`, `ShuntBank`/`MSC`/`MSR`,
  `ShuntIntegrator`; anti-windup clip, hysteresis quantiser, dwell, budget,
  feasibility guard. `g_H = h_Hᵀ·∇_y f` in nameplate-Mvar coordinate; device sign
  carried by `h_H`.
- **edit** `controller/tso_controller.py` — factored `_compute_output_gradient`
  (single source of `∇_y f`; the MIQP path's equivalence is pinned by
  `tests/test_tso_output_gradient.py`); fixed a latent `UnboundLocalError`
  (`has_pcc_rows` used before assignment for PCC-only zones).
- **edit** `controller/dso_controller.py` — `receive_disturbance_message` passes
  the shunt index to the SMW; docstring clarified (SMW-only, operating point
  held).
- **edit** `sensitivity/jacobian.py` — `apply_shunt_step_change_smw(…, shunt_idx=None)`.
- **edit** `core/message.py` — `ShuntDisturbanceMessage.shunt_indices` (optional).
- **edit** `network/ieee39/hv_networks.py`, `network/ieee39/meta.py` — `msc_msr`
  install path (MSC `q_mvar<0`, MSR `q_mvar>0`, `max_step=N`); `kind`/`n_levels`
  meta fields.
- **edit** `configs/multi_tso_config.py` — `shunt_dispatch` + integrator params +
  `validate_integrator_mode()`.
- **edit** `experiments/runners/multi_tso_dso.py` — `_shunt_mode` resolution,
  branched build, per-zone integrator construction, the atomic commit block
  (toggle + FF offset + SMW + disturbance msg), and the FF-offset injection into
  the Q_PCC setpoints sent to the DSO.
- **edit** `experiments/helpers/plant_io.py` — `apply_shunt_commit`.
- **tests** `test_shunt_integrator.py` (logic), `test_msc_msr_banks.py` (build +
  V-direction signs), `test_shunt_integrator_integration.py` (real sensitivities),
  `test_tso_output_gradient.py` (gradient consistency).
- **script** `experiments/diag_shunt_integrator.py` (end-to-end smoke).

## 5. Validation results

- **Unit** (`test_shunt_integrator.py`, 13): commit on sustained vs not on
  transient, dwell blocks chatter, feasibility guard blocks overshoot, daily
  budget caps + resets, anti-windup clamp bounds the aux state, MSC/MSR signs.
- **Build** (`test_msc_msr_banks.py`, 3): two banks/DSO with correct `q_mvar`
  signs; **MSC step raises V, MSR step lowers V**; legacy bipolar unchanged.
- **Integration** (`test_shunt_integrator_integration.py`, 4): against the REAL
  Jacobian — boundary `∂V/∂Q_eq>0` (MSC) / `<0` (MSR); interface feedforward
  finite + opposite-signed; a sustained under-voltage gradient commits one MSC
  step while a transient does not; the guard blocks under a tight band.
- **Gradient consistency** (`test_tso_output_gradient.py`, 1):
  `_compute_objective_gradient == _compute_output_gradient @ H`.
- **End-to-end smoke** (`diag_shunt_integrator.py`): 12-step integrator-mode run
  completes; 8 MSC/MSR banks built across 2 zones; no commits in baseline (EHV
  voltages near setpoint → correctly no switching).
- **Full commit path** (forced run: `v_setpoint=1.10`, guard band widened, high
  gain, `t_dwell=0`): MSC banks step up under sustained under-voltage; the DSO
  feedforward `dQ_itf` is finite and sign-consistent (≈ −38 Mvar engaging, +55
  releasing); the rank-1 SMW refresh and `ShuntDisturbanceMessage` dispatch run
  with **no power flow** and no errors.  (The chatter seen between MSC/MSR at a
  shared tertiary under those deliberately extreme settings is exactly what the
  guard + dwell + budget suppress in a tuned config.)

## 6. Open TODOs / findings

1. **Gain scale.** Boundary `∂V/∂Q_eq` at the **EHV** buses is small, so the
   default `shunt_int_alpha=0.05` is far too small to ever commit from EHV
   voltage tracking alone; it must be sized to the sensitivity magnitude. The
   gain is intuitive only relative to that. **Tune `shunt_int_alpha` per network.**
2. **Linear feasibility guard is the binding constraint — and over-conservative.**
   `v_meas + h·(s·q_step)` over-predicts the post-step voltage badly for a 50 Mvar
   step: ~2 p.u. at the local HV bus, and even the nearest **EHV** bus has
   `h ≈ 0.0125` p.u./Mvar → a +0.6 p.u. predicted jump.  So with any realistic
   band the guard blocks every commit (confirmed: commits only fire once the band
   is widened to ~[0, 100]).  The linearisation is valid only locally; a 50 Mvar
   step is far outside it.  **This must be fixed before the feature is usable** —
   e.g. clamp the predicted ΔV, attenuate by an empirical large-step factor, or
   evaluate the guard only at the stiff EHV boundary with a realistic ΔV cap.
   (Design change — flagged for discussion, not changed unilaterally.)
3. **Reserve term not projected onto the shunt.** The integrator's `g_H` uses the
   voltage-tracking and `Q_PCC` terms (clean boundary sensitivities). The SG
   reactive-reserve term (`g_res_sg`) is in `∇_y f` but there is no
   `∂Q_gen/∂Q_shunt` helper yet, so it does not (yet) drive the shunt. Adding it
   would strengthen the "offload generators on sustained need" behaviour.
4. **`local_sensitivities_tso=True` unsupported** in integrator mode — the
   tertiary bus is dropped under the reduction, so the boundary column cannot be
   identified there. The runner raises `NotImplementedError`. Generalise via the
   existing synthetic-shunt-at-3W-primary remapping if local-net TSO is needed.
5. **`dQ_itf_sh` sign assert** is currently finiteness-only; the full
   sign-consistency-with-susceptance assert is deferred — the intended end-to-end
   check is the "DSO DER does not lunge to cancel" behaviour.
6. **Validation scenarios 3 & 4** from the plan (feedforward ON: DSO DER ≈
   unchanged + TSO gens de-load one step; feedforward OFF: DSO DER compensate)
   need a sustained-stress scenario + per-step record introspection — deferred to
   a tuned scenario run.

---

## Follow-up (2026-06-22): units bug, local-mode support, reserve term, g_w

Four follow-up items addressed (user request).

### (2) Sensitivity units bug — `∂V/∂Q` was ~100× too large [FIXED in integrator]
Finite-difference validation against `pp.runpp` showed `compute_dV_dQ_der` /
`compute_dV_dQ_shunt` return values **`sn_mva` (= 100) times too large**: the
reduced Jacobian is `∂V[pu]/∂Q[pu]` on the 100 MVA base, but it is consumed as
`pu/Mvar` without the `/sn_mva` conversion. True `∂V/∂Q` at the 345 kV buses is
~1.3e‑5 pu/Mvar (correct for a stiff EHV node), not the 0.0125 reported earlier.
The **interface‑Q** (`compute_dQtrafo3w_hv_dQ_shunt`, FD ratio 1.07) and
**`∂Q_gen/∂Q`** sensitivities are *correct* — they are dimensionless Mvar/Mvar
ratios, so the base cancels.
- *Fix applied:* the integrator divides `∂V/∂Q` by `s_base` at point of use
  (runner), so its gradient term and the overshoot guard are in physical units.
- *This resolves finding #2 above:* with correct units the guard's predicted
  step is ~`1.3e‑5 × 50 ≈ 7e‑4` p.u. (physical), so it no longer blocks every
  commit. The earlier "guard over‑predicts" symptom was mostly this 100× bug;
  the residual concave‑`V(Q)` nonlinearity over a 50 Mvar step is second‑order.
- **Open (systemic, needs your decision):** the *same* 100× scale is in the
  shared `dV_dQ_reduced`‑based `∂V/∂Q` used by the **MIQP** H (voltage rows for
  Q‑actuators). The MIQP masks it because a constant scale is absorbed into the
  tuned `g_w`, but it is physically wrong and affects the voltage‑constraint
  predictions. A global fix (`/sn_mva` at the source) is correct but would
  invalidate all current `g_w` tuning — **left untouched pending your call**; I
  only corrected the (new, untuned) integrator path.

### (1) Integrator now works with `local_sensitivities_tso=True` [DONE]
Removed the `NotImplementedError`. The per‑zone reduced net keeps the PCC 3W
couplers + their tertiary buses + the TSO‑owned tertiary shunts
([network_reduction.py:287](sensitivity/network_reduction.py:287)), so the
integrator's `compute_dV_dQ_shunt` / `compute_dQtrafo3w_hv_dQ_shunt` and the
rank‑1 SMW refresh all operate on the per‑zone reduced Jacobian
(`tso_ctrl.sensitivities`) with the tertiary bus + shunt index intact —
**still no power flow on a switch**. Validated end‑to‑end:
`diag_shunt_integrator.py` with `local_sensitivities_tso=local_sensitivities_dso=True`
builds 8 banks and commits across zones with no error.

### (3) `shunt_int_alpha` → `shunt_int_g_w` [DONE]
The integrator no longer uses a standalone `alpha`. Consistent with the rest of
the controller (alpha fixed = 1, step amplitude tuned via `g_w`), the relaxation
advances by `Δ = g_H / (2·g_w)`. Config field `shunt_int_g_w`; `ShuntBankConfig.g_w`.
NOTE: because the boundary gradient is small (correct units), `g_w` must be small
(≪ 1) to commit — tuning item.

### (4) `∂Q_gen/∂Q_shunt` wired so the reserve term drives the shunt [DONE]
No new derivation needed — `JacobianSensitivities.compute_dQgen_dQ_shunt_matrix`
already exists (dimensionless Mvar/Mvar × `q_step·V²`, base‑correct). The
integrator now adds the SG reactive‑reserve term `grad_y_Qgen · ∂Q_gen/∂Q_eq` to
`g_H`, **opt‑in via `tso_g_res_sg ≠ 0`**. Validated: a run with `tso_g_res_sg>0`
completes and commits without error. Activate it to let the bulk shunt offload
generator reactive loading on sustained need.

Full non‑tuning suite after these changes: **381 passed, 11 skipped, 0 failures.**

---

## Follow-up 2 (2026-06-22): units bug fixed at source (option A)

The `∂V/∂Q` 100× error is now fixed **at the source**, not just in the integrator.

**Root cause (confirmed by FD):** `dV_dQ_reduced` is `∂V[pu]/∂Q[pu]` on the
100 MVA base; `compute_dV_dQ_der` returned it without the `/sn_mva` conversion
(unlike `compute_dI_dQ_der`, which divides). Callers compensated *inconsistently*
— so the MIQP H matrix was internally inconsistent (voltage rows 100× larger
than the correct current / interface‑Q / Q_gen rows).

**Fix:**
- `compute_dV_dQ_der` now divides by `net.sn_mva` → returns physical `[pu/Mvar]`
  (cascades to `compute_dV_dQ_shunt`, the MIQP H V‑rows, and cross‑zone coupling).
- Removed the now‑redundant `/sn_mva` at the five sites that compensated locally:
  `dso_controller` ×2 (w‑shift T′), `tso_controller` (w‑shift T′),
  `der_qv_local_loop` ×2 (S_VQ seed / svq column), and the integrator's local
  `/s_base`. (Leaving them would double‑divide → ÷10 000.)
- `compute_dI_dQ_der` and the dimensionless `∂Q/∂Q` ratios were already correct
  and were left untouched.
- `tests/test_jacobian.py::test_dV_dQ_der_numerical_agreement` now asserts
  analytical ≈ numerical (ratio 0.99999) instead of the old "ratio ≈ sn_mva"
  bug‑encoding.

**Re‑tuning required (consequence of the fix):** the MIQP voltage‑tracking
gradient now scales 100× smaller (correct units), so to preserve dispatch
behaviour **multiply `g_v` and `dso_g_v` by ~100 in every experiment config**
(e.g. `g_v 3e5 → 3e7`, `dso_g_v 15000 → 1.5e6`). All other weights are
unchanged (Q_PCC/Q_tie/Q_gen rows are dimensionless ratios; current rows were
already correct). The voltage **constraint** is now physically correct (was
100× over‑conservative), so re‑validate rather than trusting the mechanical ×100.

Suite after the source fix: **381 passed, 11 skipped, 0 failures**; the
integrator still commits end‑to‑end under local sensitivities.

### TSO live-plot: integrator shunts now shown (was "no shunts in network")
The TSO controller live plot counted shunts from the MIQP config
(`zd.shunt_bus_indices`, empty in integrator mode) and read states from the MIQP
solution — so it reported "no shunts" even while banks were committing. Fixed:
- the bank **count** is taken from `meta.tso_tertiary_shunt_zones` in integrator
  mode ([multi_tso_dso.py:1891](experiments/runners/multi_tso_dso.py));
- the per-step **states** are read from the integrator banks' committed
  pandapower steps (by explicit shunt index), and **signed by device class** to
  match the legacy bipolar convention: **MSR (reactor) positive, MSC (capacitor)
  negative**, with a zero baseline in the tile so the two classes are visually
  separated.

### Integrator tuning quick-reference
- `shunt_int_delta_mvar` (δ): hysteresis half-width [Mvar]; commit thresholds are
  at `q_step/2 ± δ`, giving a 2δ anti-chatter dead-band. `0 < δ < q_step/2`.
- `shunt_int_g_w`: step weight; aux advances `g_H/(2·g_w)` per TSO iteration, so
  iterations-to-commit ≈ `2·g_w·(q_step/2+δ)/|g_H|`. Smaller → eager, larger →
  integrating/selective. Scales **with `g_v`** (×100 g_v ⇒ ×100 g_w to hold
  cadence). Tune by watching `[shunt-commit]` cadence.
- `g_w_tso_shunt`: MIQP shunt change-penalty — **ignored in integrator mode**
  (shunt not in the MIQP); integrator analogue is `shunt_int_g_w`.
