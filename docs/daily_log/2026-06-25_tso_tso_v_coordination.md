# 2026-06-25 — Horizontal TSO–TSO coordination (V-corridor + Q_tie soft cap)

**Author:** Manuel Schwenke (with Claude Code)
**Scope:** Add a *horizontal* coordination layer between peer TSO zones across
their shared tie lines, alongside the existing *vertical* TSO–DSO cascade. The
coordinated variable is the **boundary-bus voltage** (corridor reference, the
primal), trimmed by a **single consistency price per tie** (the lightweight
dual). The inter-zone reactive *exchange* is bounded separately and locally by a
**Q_tie soft cap**. Minimal scope as agreed: message type + coordinator +
zone hooks. Runner wiring intentionally deferred (see "Not done").

## Rationale (why V, not Q_tie, as the coordination variable)

Voltage is the directly controllable, locally observable primitive (`∂V/∂u`
strong); tie-line reactive flow is jointly determined and only weakly
controllable (`∂Q_tie/∂u` acts through `V_i − V_j`, with an unobservable
common-mode null space if both zones raise V together). Driving the two
endpoints to a common corridor voltage parks the exchange at its irreducible,
active-flow-driven minimum — subsidiarity achieved physically. The exchange
*band* is then a guardrail, enforced as a soft cap on the `Q_tie` output, not a
tracked target. Line drop (`ΔV` between endpoints) is obtained by subtracting
the two boundary measurements (not modelled), held constant within a round as a
feedforward, since it is dominated by the exogenous active transfer.

## Per-round update (one outer Layer-1 round, per tie e = (i, j))

```
c_i   = V_i − κ·dv_ff                # zone-i implied corridor (drop-corrected)
c_j   = V_j + (1−κ)·dv_ff            # zone-j implied corridor
vbar  = ½·(c_i + c_j)                # joint-preferred corridor voltage
r     = (V_i − V_j) − dv_ff          # disagreement beyond the fed-forward drop
lam   ← clip(lam + α_λ·r, ±lam_max)             # dual ascent (consistency)
v_ref ← clip(v_nom + DB_Δv(vbar − v_nom), [v_min, v_max])   # primal + subsidiarity
dv_ff ← (1−β)·dv_ff + β·(V_i − V_j)             # refresh feedforward
```
`DB_Δ(x) = sign(x)·max(0, |x|−Δ)` is the deadband soft-threshold (prox of the
subsidiarity penalty). Messages carry per-end setpoints `V_ref_i = v_ref+κ·dv_ff`,
`V_ref_j = v_ref−(1−κ)·dv_ff` (so `V_ref_i − V_ref_j = dv_ff`, matching the
measured drop — zones are not asked to fight the physical drop) and signed
prices `+lam` / `−lam`.

## What was added

**New module `controller/tie_coordinator.py`** (pure, plant-free, unit-testable):
- `TieLink` — static per-tie description (fixed `i`/`j` orientation = price sign,
  the two zone controller ids + boundary buses, `v_nom`).
- `TieCoordinatorConfig` — `deadband_v_pu`, `alpha_lambda`, `lambda_max`,
  `v_min/​max_pu`, `kappa`, `ff_smoothing`; validated. Conservative defaults;
  `alpha_lambda`/`lambda_max` must be tuned relative to `g_v` (price is on the
  same scale as `2·g_v·Δv`).
- `HorizontalTieCoordinator` — holds `(v_ref, lam, dv_ff)` per tie; `update(realized)`
  advances state from realised boundary voltages; `generate_messages()` emits one
  aggregated `TieCoordinationMessage` per zone; `state()` for logging.

**New message `core.message.TieCoordinationMessage`** (coordinator → zone):
`tie_line_indices`, `boundary_bus_indices`, `v_ref_pu`, `price` (signed), with
length-consistency validation. Mirrors the `SetpointMessage` pattern.

**`controller/tso_controller.py` hooks:**
- `TSOControllerConfig.q_tie_band_mvar` (Optional, per-tie ≥ 0). When set,
  `_get_output_limits` tightens the `Q_tie` bound from wide-open to `[−band, +band]`,
  enforced as a soft slack via `g_z_q_tie` (which must be > 0 to bite; with it 0
  the band gets free slack and is inert — safe).
- `receive_tie_coordination(message)` — redirects `v_setpoints_pu` at each
  boundary bus to the corridor reference and stores the signed price keyed by the
  bus's row in `voltage_bus_indices` (`_tie_price_by_vrow`, replaced each round).
  Raises on wrong target, unknown bus, or disabled voltage tracking.
- `report_tie_boundary_voltage(measurement, bus)` — measured V for the
  coordinator's realised-voltage input.
- **Price injection** added to *both* `_compute_output_gradient` (constant
  `s_k·λ` on the boundary bus's voltage row of `∇_y f`, no factor 2 / `g_v`) and
  `_compute_objective_gradient` (`Σ price·H[v_row,:]`), so the invariant
  `∇_u f == ∇_y f @ H` (relied on by the shunt integrator) is preserved exactly.

## Key structural point

The only genuinely new *local* objective term is the scalar price; the
consensus-tracking term is the **existing** voltage tracking with its boundary-bus
setpoint redirected. Everything is guarded (`if self._tie_price_by_vrow` /
`if q_tie_band_mvar is not None`) so the baseline (no coordination) path is
byte-for-byte unchanged.

## Tests

- `tests/test_tie_coordinator.py` (21) — deadband, fixed point at nominal, joint
  preference vs. anchor, dual ascent/clipping, feedforward absorbing the steady
  drop, reference clipping, per-end split + signed price, multi-tie aggregation,
  config/message validation.
- `tests/test_tie_coordination_hooks.py` (9) — setpoint redirect + price store,
  bad-target/unknown-bus rejection, price-map replacement, **gradient invariant
  preserved with non-zero price**, zero-price == baseline, Q_tie band tightening
  / wide-open default, boundary-voltage report.
- Regression: `tests/test_tso_output_gradient.py` and `tests/test_controller.py`
  (56 passed, 1 skipped) unchanged.

## Part 2 — Runner wiring + live plot (same day)

**Runner** (`experiments/runners/multi_tso_dso.py`, all gated on
`config.enable_tie_coordination`; no-op when off):
- After `tie_line_map` is built, construct one `TieLink` per inter-zone tie line
  (orientation `zone_i < zone_j`, endpoints from the existing
  `ZoneDefinition.tie_line_endpoint_buses`), and **extend each zone's
  `v_bus_indices`** with its tie endpoint buses *before* controller construction
  so the corridor setpoint + price act on real V rows (and the downstream H /
  g_z / v_setpoint sizing pick them up).
- Pass `q_tie_band_mvar = full(n_tie, config.tie_q_band_mvar)` into each zone's
  `TSOControllerConfig`.
- Build the `HorizontalTieCoordinator` after the controllers exist; warn if
  `g_z_q_tie <= 0` (cap inert).
- **Horizontal round** inserted right after the per-zone measurements are taken
  and *before* `coordinator.step` (the TSO MIQP solve): collect realised
  boundary voltages via `report_tie_boundary_voltage`, `coordinator.update`,
  then deliver `generate_messages()` to each zone via `receive_tie_coordination`.
  Records `tie_lambda / tie_v_ref / tie_dv_ff / tie_v_i / tie_v_j / tie_q_mvar`
  (per tie id) into the step record.

**Config** (`configs/multi_tso_config.py`): `enable_tie_coordination`,
`tie_deadband_v_pu`, `tie_alpha_lambda`, `tie_lambda_max`, `tie_v_nom_pu`,
`tie_v_min_pu`, `tie_v_max_pu`, `tie_kappa`, `tie_ff_smoothing`,
`tie_q_band_mvar`, and `live_plot_tie_coordination`.

**Record** (`experiments/helpers/records.py`): six per-tie dicts added to
`MultiTSOIterationRecord` (`tie_lambda`, `tie_v_ref`, `tie_dv_ff`, `tie_v_i`,
`tie_v_j`, `tie_q_mvar`), defaulted so old pickles still load.

**Live plot** (new `visualisation/plot_tie_coordination.py`,
`TieCoordinationLivePlotter`, Figure 5, gated on
`config.live_plot_tie_coordination`): 4 tiles per tie line — (1) consistency
dual λ_e, (2) tie-line reactive flow with the ±`tie_q_band_mvar` soft cap
shaded, (3) boundary voltages V_i (solid) / V_j (dashed) / corridor v_ref
(dotted) vs v_nom, (4) feedforward drop dv_ff. Mirrors the
`TSOControllerLivePlotter` style helpers. Note: `tile_title` / section bands
upper-case their text, which corrupts `$…$` mathtext (`\mathrm`→`\MATHRM`), so
titles are plain text and mathtext lives only in y-labels / legend handles. The
plotter skips DSO-only records (empty `tie_lambda`) so every trace shares a
gap-free TSO-step time axis.

**Verification:** 20-min CIGRE smoke with coordination on found 5 tie lines
`[2, 5, 14, 18, 25]`, populated the coordinator state every TSO step (λ small,
v_ref pulled toward nominal under the deadband, `dv_ff = V_i − V_j`, per-line Q
recorded); headless render of the plotter produced the expected 4-tile figure;
`tests/test_tie_coordinator.py` + `test_tie_coordination_hooks.py` +
`test_tso_output_gradient.py` (31) green.

> **Default flip (by the user/linter):** `enable_tie_coordination` now defaults
> **True** and `tie_q_band_mvar` to **10**. Consequence: every *multi-zone* run
> now gets V-coordination (endpoints added to `v_bus_indices`, corridor + price
> applied), changing results vs. the pre-coordination baseline. With the default
> `g_z_q_tie = 0` the Q_tie cap is inert (free slack) and a one-time warning
> fires. Set `enable_tie_coordination=False` to reproduce baselines, or
> `g_z_q_tie>0` to actually enforce the band. Central (V5) / local-TSO runs skip
> the horizontal round (the TSO-step gate), though endpoints are still added to
> the monitored voltage set.

## Open / risks

- Constant-drop feedforward assumes active transfer ~steady over a round;
  fast-ramping ties need more frequent refresh.
- Dual stability is local (integers fixed in round); `α_λ`/`lam_max` set the
  inter-controller loop gain — a contraction check should precede any
  non-oscillation claim.
- `κ = 0.5` recommended; the consensus estimate is exact for any `κ` but the
  per-end split is arbitrary otherwise.
