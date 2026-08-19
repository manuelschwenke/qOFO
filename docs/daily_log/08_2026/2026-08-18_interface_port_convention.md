# 2026-08-18 — Interface port convention: primary vs secondary, measured; and the delegated-variable redefinition

**Author:** Manuel Schwenke / Claude Code
**Timestamp:** 2026-08-18, Europe/Berlin
**Reason:** Design review of how a TS zone models its subordinate DS and its
dispatch — specifically whether the subordinate controller should regulate
reactive power at the coupler's **primary (EHV)** port, as implemented, or at
its **secondary (110 kV)** port. No code was changed.

## What the implementation does

| element | where |
|---|---|
| frozen PQ equivalent of the DS at the **MV bus** (coupler kept alive, HV sub-net dropped) | `sensitivity/network_reduction.py:660-700`, load created at `:880-882` |
| virtual actuator `Q_PCC,set` at the **primary bus** — identity on the Q_PCC row plus `-dV/dQ_inj`, `-dI/dQ_inj` at `hv_bus` | `controller/tso_controller.py:1925-1975` |
| tertiary bank as a TSO actuator, DSO blind to it as a control variable | `experiments/runners/multi_tso_dso.py:895-913`, DSO gets `shunt_bus_indices=[]` at `:1354` |
| tracked variable on **both** layers = `res_trafo3w.q_hv_mvar` (EHV port) | `core/measurement.py:275-277`, `:331-337` |
| DSO actuators: DER Q + the coupler's own tap (physically on the HV winding) | `network/ieee39/hv_networks.py:207-212` |

Two stale artefacts found in `network_reduction.py`: the module docstring
(`:30-38`) still describes the earlier "PQ at the primary, coupler dropped"
design, and `primary_load_specs` (`:624-631`) is computed and never consumed.

## Measured (this is the content of the design question)

Plant: IEEE 39 + four 110 kV sub-networks, `rural_700`, 2016-01-05 08:00,
4.76 GW, `run_control=False`, central finite differences. Couplers lightly
loaded (S/Sn 0.03–0.24). Probes in the session scratchpad, not promoted.

| quantity | at HV port (implemented) | at MV port (alternative) |
|---|---|---|
| coupler Q consumption (bias term) | inside the DSO's problem: 0.14–1.79 Mvar/trafo, 2.2 Mvar/area | moves to the TSO, same magnitude, scales (S/Sn)² |
| marginal port gain `dq_hv/dq_mv` | 1.000 by definition | 0.989–0.992, checked to ×7 DS load |
| MSC step (≈54 Mvar), host trafo | −53.9 Mvar | −11.8 Mvar |
| MSC step, siblings | −6.3 / −5.6 | +6.3 / +5.7 |
| **MSC step, netted over the area** | **−65.8 Mvar** | **+0.18 Mvar** |
| coupler tap ±1 | −4.2 … −9.6 | +4.3 … +9.8 |
| bank step → DSO's own 110 kV buses | 0.7–3.3 pu-% | unchanged |

## Decision: keep the primary-side convention

Reasons, in order of weight:

1. **The identity column `dQ_PCC/dQ_PCC,set = 1` is exact only under this
   convention**, because the commanded variable *is* the wanted variable. Under
   an MV convention the TSO's model would need 1.011 plus a bias equal to the
   coupler's Q consumption (2.2 Mvar/area here, `vk_mv * Sn = 24 Mvar` per
   transformer at rated loading) — a bias on the *supervisory* controlled
   output, in the same band as the residuals of 2026-08-13 (+1.7 / −7.3 Mvar).
2. **Model error should sit where the model is good.** The DSO keeps the coupler
   as a real 3W element with the true tap; the TSO's view of the DS is a frozen
   PQ.
3. **Settlement co-location.** SBX-V is defined on `q_hv_mvar` at the EHV port
   (DP3, confirmed 2026-07-09, `docs/status/STATUS_SBXV.md:485`).
4. **The claimed gain is smaller than it looks.** The MV invariance is a
   *netted-area* property (+0.18 of 54 Mvar), not per-transformer (−11.8 at the
   host); the disturbance channel survives either way (0.7–3.3 pu-% on the DSO's
   own buses); and the invariance rests on the constant-PQ / constant-Q DS
   model, degrading with an active, non-re-anchored Q(V) layer.

## Correction to an argument made earlier in the session

An earlier draft argued that the primary-side convention makes the shunt column
and the `Q_PCC,set` identity column *exact substitutes* in the TSO's Q_PCC row,
leaving the choice to the change penalties. **That is void for the shipped
configuration.** In `shunt_dispatch="integrator"` the per-zone shunt lists are
empty (`:899-913`), so `H` carries no shunt column at all; the banks are driven
by the separate `ShuntIntegrator`, whose own scalar gradient carries the
interface term via `compute_dQtrafo3w_hv_dQ_shunt`. The two mechanisms are
sequential with different authority and rate, not collinear in one program. The
collinearity argument applies only to the legacy `shunt_dispatch="miqp"` path.

## Adopted: the delegated-variable redefinition (prose only)

Define `qtilde = q_hv - Qsh`, with `Qsh` the estimated contribution of
TSO-owned tertiary compensation to the measured primary-port flow. The DSO
tracks `qtilde` against `u_pcc`. This is algebraically what the code already
does — the setpoint feedforward and the capability-band shift are the same
change of variable applied in two places — so it is a dissertation change with
no code implication. Full instruction, identities with code anchors, and the
list of claims that must **not** be made:
`docs/handoff/2026-08-18_interface_variable_redefinition.md`.

## Risks / unresolved

- One operating point, no controllers in the loop; the loading sweep reached
  only S/Sn 0.45, so the quadratic regime of the loss term is extrapolated from
  `Q_loss = vk_mv * Sn * (S/Sn)^2` (which reproduced the measured 1.79 Mvar to
  within 25 %).
- The droop-active channel was not measured — the bare-built net carries no
  `QVLocalLoop` controllers, so `run_control=True` was a no-op.
- `build_tso_local_net` keeps the real tertiary shunt (`:717`) *and* creates a
  synthetic primary-bus copy (`:886`). Inert in integrator mode (zone shunt
  lists empty) and at step 0, but in legacy `miqp` mode with a non-zero cached
  step the reduced net would carry the susceptance twice. Not fixed.
