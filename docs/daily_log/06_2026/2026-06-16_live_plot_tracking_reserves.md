# 2026-06-16 — New live plot: TRACKING ERRORS & RESERVES

**Timestamp:** 2026-06-16 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
The existing three live figures (MULTI-TSO CONTROLLER, CASCADE-DSO,
SYSTEM POWER FLOW) show raw measured / actuated quantities. There was no
single view of *control-performance KPIs*: how well each control objective is
tracked and how much reactive reserve the continuous actuators retain. This
adds a fourth, togglable live figure exposing six such KPIs, activated like
the others via a config flag (`live_plot_tracking=True`).

**Scope (per user decision):** wired into the multi-zone runner
`run_multi_tso_dso` only — i.e. experiments 000 / 002 / 003 / 005 (and 006).
**001** (`run_cascade`, single-TSO/single-DSO, different `IterationRecord`
schema) is intentionally **not** covered: tie-line and multi-zone metrics are
degenerate there.

## What the figure shows (6 tiles, 2 bands)

TRACKING ERRORS
1. **TS voltage tracking error per zone** — spatial RMS of `(V − V_set)` over
   each zone's observed EHV buses (`zd.v_bus_indices`); one line per zone.
   Source: existing `rec.zone_v_rms_err_pu[z]`.
2. **TS voltage tracking error (system RMS)** — single aggregate line, the
   **bus-count-weighted** combination of the per-zone RMS values:
   `sqrt( Σ_z n_z·rms_z² / Σ_z n_z )` with `n_z = len(zd.v_bus_indices)`
   (= true RMS over all observed EHV buses system-wide). Computed in the
   plotter from per-zone RMS + static bus counts.
3. **TSO–DSO interface Q tracking error per DSO** — one line per DSO,
   `RMS_over_its_trafos(Q_actual − Q_set)`. Source: existing
   `rec.dso_trafo_q_actual_mvar` / `rec.dso_trafo_q_set_mvar`, keyed
   `"{dso_id}|trafo_{idx}"`.
4. **Tie-line Q tracking error (system RMS)** — single aggregate line,
   `RMS_over_tie_pairs(Q_tie − Q_tie_set)`. Source: existing
   `rec.zone_tie_q_mvar[(zi,zj)]`. Reference is the configured tie setpoint,
   which currently defaults to **0 Mvar** (`ZoneDefinition.q_tie_setpoints_mvar`
   is all-zeros), so the plotter's default-0 reference is used.
5. **Synchronous-generator Q reserve** — one line per SG,
   `r_Q = min(Q_max−Q, Q−Q_min) / (Q_max−Q_min)` with the band from the
   Milano §12.2.1 capability curve (`ActuatorBounds.compute_gen_q_bounds`,
   P- and V-dependent). New record field `gen_q_reserve`.
6. **TSO-DER Q reserve** — one line per TSO-connected DER, same `r_Q`
   definition, band from `ActuatorBounds.compute_der_q_bounds`
   (VDE-AR-N-4120 / STATCOM, P-dependent). New record field
   `tso_der_q_reserve`.

`r_Q` ∈ [0, 0.5]: 0 at a capability limit (no reserve), 0.5 mid-band
(maximum reserve); dips below 0 if realised Q leaves the operating-point
band. **NaN** where the band has zero width (e.g. DER in the VDE dead zone
`P/S_n < 0.1`, or `Q_max == Q_min`) — guarded by a `> 1e-9 Mvar` width test.

## Method / where each KPI is produced
- **Voltage RMS (tiles 1–2):** already recorded; tile 2's aggregate is
  reconstructed exactly from per-zone RMS because the per-zone bus counts are
  static (no record change needed).
- **Interface-Q / tie-Q RMS (tiles 3–4):** computed inside the plotter each
  `update()` from per-trafo / per-pair record dicts (consistent with how the
  CASCADE-DSO plot aggregates). No record change.
- **Reserves (tiles 5–6):** computed in the runner where the capability
  bounds + P/Q/V are already in hand, then stored as per-zone arrays so the
  plotter just draws them. SG reserve reuses the `q_min_cap/q_max_cap` already
  computed for `gen_q_headroom_mvar`; DER reserve calls
  `compute_der_q_bounds` on the same `zd.tso_der_indices` order used for
  `zone_q_der`.

All six KPIs derive from quantities recorded **every step** (not gated on
`tso_active`), so the plot updates smoothly each step like the others.

## What was changed

### `visualisation/plot_tracking.py` (new)
- `TrackingLivePlotter` — 6-tile / 2-band figure mirroring
  `TSOControllerLivePlotter` (same `style.py` header/band/colour helpers,
  GridSpec layout, shared-x cleanup, Qt slot placement). Constructor takes
  `zone_ids`, `n_v_bus_per_zone`, `dso_ids`, `dso_trafo_keys`,
  `tie_line_pairs` (+ optional `tie_setpoints_mvar`). Empty DSO / tie / element
  sets render an italic "no …" placeholder.

### `configs/multi_tso_config.py`
- New field `live_plot_tracking: bool = False` (next to the other
  `live_plot_*` flags) with docstring.

### `experiments/helpers/records.py`
- `MultiTSOIterationRecord`: two new fields `gen_q_reserve` and
  `tso_der_q_reserve` (`Dict[int, NDArray]`, default-empty so older pickles
  still load), documented.

### `experiments/runners/multi_tso_dso.py`
- Per-zone recording block: compute `rec.gen_q_reserve[z]` (right after the
  existing `gen_q_headroom_mvar`) and `rec.tso_der_q_reserve[z]` (right after
  `zone_q_der`), both NaN-guarded for zero-width bands.
- Plotter lifecycle: `_plotter_track` created when `config.live_plot_tracking`
  is set (builds the per-DSO trafo-key map the same way the SYSTEM plot builds
  `_interface_trafo_ids`, OFO-controller form with local-HV fallback), and
  `_plotter_track.update(rec)` in the per-step update block alongside the
  other three plotters. Placed in `slot_idx=2`.

## Validation
- `py_compile` clean on all four modules.
- **Plotter unit smoke (Agg):** synthetic records over 8 steps render all 6
  tiles, both bands, legends; an empty-DSO/empty-tie/single-zone instance also
  renders (placeholders shown). Saved PNG inspected — layout matches house
  style.
- **End-to-end (Agg, `MPLBACKEND=Agg`, no Qt windows):** 4-min
  `run_multi_tso_dso` (`scenario="wind_replace"`, `live_plot_tracking=True`)
  returns 4 records; `gen_q_reserve` and `tso_der_q_reserve` populated for
  all zones (1,2,3). Samples: zone-1 SG reserve `[0.260, 0.304]`, DER reserve
  `[0.0, 0.379]` (the `0.0` = element at a capability limit), voltage RMS
  `{1: 0.00705, 2: 0.01439, 3: 0.0127}`. No exceptions; only the harmless
  Agg "non-interactive" warning from `plt.pause`.

## Open / next
- **Window placement:** the new figure uses `slot_idx=2`, which in the
  `dual_screen` layout is the secondary screen — the same slot as the SYSTEM
  POWER FLOW figure. Enabling `live_plot_system` and `live_plot_tracking`
  together will overlap their windows. If both are commonly used at once,
  extend `position_figure_in_slot` to a 4-slot scheme.
- **Tie-Q reference:** tracking error tile 4 currently assumes a 0-Mvar tie
  setpoint (matches the all-zero `q_tie_setpoints_mvar` default). Wire a
  non-zero per-pair reference into `TrackingLivePlotter(tie_setpoints_mvar=…)`
  if/when commanded inter-zone tie exchange is introduced.
- **DSO-DER reserve** is not shown (the figure covers TSO-connected DER only,
  per the request). The per-DSO DER capability is already on the record
  (`dso_group_q_der_min/max_mvar`) if a 7th tile is later wanted.
