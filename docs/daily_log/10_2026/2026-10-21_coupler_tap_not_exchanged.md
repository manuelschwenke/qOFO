# Coupler tap position is not exchanged upward — audit of the claim

- Timestamp: 2026-10-21 (CEST).
- Reason: Thesis text states "The tap position is not exchanged. The tap is
  operated only by the STS-OFO controller, which already knows its current
  position. The parent model keeps the tap fixed at its value at the
  linearisation point." Author asked whether this is a defect to be corrected
  in the controller implementation, i.e. whether the EHV/HV coupler tap should
  be communicated to the TS controller's model.
- Outcome: **no code change.** The claim is accurate for the implementation and
  structurally sound. One adjacent gap identified (capability width), logged
  below as an open point.

## What the implementation actually does

| Element | Behaviour |
|---|---|
| `sensitivity/jacobian.py` `JacobianSensitivities.__init__` | `self.net = copy.deepcopy(net)`. The controller model is a frozen snapshot; `net.trafo3w.tap_pos` is frozen with it. |
| `controller/tso_controller.py:1836` `_build_sensitivity_matrix` | H built once from `self.sensitivities.net`, cached in `_H_cache`. Rebuild triggers: outage-mask change, AVR saturation-mode change, explicit `invalidate_sensitivity_cache()`. No tap path. |
| `controller/tso_controller.py:2714` `step` | Only state-dependent refresh is the shunt `V²` column rescaling (`SensitivityUpdater`). |
| `sensitivity/index_helper.py:428` | τ = 1 + s₀·Δτ with s₀ = `tap_pos` of the frozen model — this is where the stale tap enters the parent's 3W partials. |
| `sensitivity/network_reduction.py:642-700` (`local_sensitivities_tso=True`, the Ch-9 selected design) | The 3W coupler is kept alive, but the HV sub-network behind it is replaced by a constant PQ load at the MV bus. |
| `configs/config.py:938` `refresh_shared_jac_on_tso = False` | The re-linearise-every-TS-tick mode exists but is off by default and explicitly a no-op in local-sensitivity mode. |
| `controller/dso_controller.py:406` `receive_disturbance_message` | Precedent for the inverse direction: a TSO-owned shunt step inside the DSO net is announced and absorbed by a rank-1 Sherman–Morrison update of the child's cached Jacobian, with no `pp.runpp` ("controllers never see the plant"). No tap analogue exists. |

## Why the frozen tap is not a first-order error

`controller/tso_controller.py:2026` sets `H[q_pcc_row + j, n_der + j] = 1.0`:
the parent's ∂Q_PCC,i/∂Q_PCC,set,i is a **hard-coded identity**, not a Jacobian
quantity. It is therefore tap-independent by construction. Combined with
`apply_avt_reset` (PCC-Q block of `_u_current` re-anchored on measured
`interface_q_hv_side_mvar` each tick) and `_compute_input_bounds`
(`u_bound = Q_PCC,measured + capability_delta`), the tap's realised effect
reaches the parent through the measurement, not through the model. The tap is
a *parameter of H*, not a state the parent must hold.

## Measurement (probe, scratchpad only — not committed)

Full `build_ieee39_net("base_410") + add_hv_networks`, plain
`pp.runpp(numba=False)` (not the project's ZIP / `enforce_q_lims` /
distributed-slack options), coupler `trafo3w 0` (DSO_1, primary bus 6),
1.25 %/tap, range ±13. Parent's stale H at s₀ = 0 vs re-linearised H at s₀+k:

| k | τ | plant ΔQ_hv [Mvar] | ∂V_prim/∂Q_prim error | max abs error in ∂Q_PCC/∂V_gen [Mvar/pu] |
|---|---|---|---|---|
| 1 | 1.0125 | −6.10 | 0.02 % | 0.058 |
| 2 | 1.0250 | −11.90 | 0.04 % | 0.113 |
| 3 | 1.0375 | −17.44 | 0.06 % | 0.165 |
| 6 | 1.0750 | −32.52 | 0.11 % | 0.308 |
| 13 | 1.1625 | −60.30 | 0.24 % | 0.571 |

Readings:

- The V-row block the TS voltage-tracking objective uses for the Q_PCC,set
  columns (`compute_dV_dQ_der` at the EHV primary bus) is numerically
  tap-independent: 0.24 % over the **entire** ±13-tap range.
- One tap step moves the controlled output by 6.1 Mvar while corrupting the
  parent's Q_PCC-row gradient by 0.058 Mvar/pu. Two orders of magnitude
  separate the measured first-order effect from the model's gain error.
- The relative error of the Q_PCC rows is large (82–98 % in row norm) only
  because the base value is near zero at the linearisation tap; the absolute
  stakes are small.
- Upper bound only: in the as-run configuration the HV network behind the
  coupler is a constant-PQ Ward equivalent, so its voltage-dependent
  contribution — the dominant physical tap effect — is absent from the
  parent's model regardless of the tap value.

## Open point (separate from the thesis claim)

`controller/dso_controller.py:469` `generate_capability_message` builds
`q_min_mvar` / `q_max_mvar` **from the DER Q rails only**
(`compute_der_q_bounds` × open-loop ∂Q_iface/∂Q_DER). The coupler tap's
remaining headroom is not in the reported envelope, and
`controller/multi_tso_coordinator.py` does not augment it. The *anchor* of the
parent's PCC bound is the measured interface Q, so the tap's achieved
contribution is included; only the *width* is under-reported. Direction of the
error is conservative (the parent under-uses the child), but it is a genuine
deficit in the upward channel and concerns range, not the tap integer.

## If the H staleness is ever to be closed

Do **not** ship the tap integer and re-linearise — that requires a power flow
on the parent's model net and reintroduces plant visibility. The consistent
route is the mirror image of `ShuntDisturbanceMessage`: an upward tap-change
notification absorbed by a rank update of the parent's cached Jacobian at the
coupler branch, holding `(V, θ)` fixed. Architectural change; not implemented.
