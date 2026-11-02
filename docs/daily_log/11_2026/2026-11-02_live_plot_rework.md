# Live plots reworked: TS / STS / overview windows, plant-level objective Φ recorded

**Date:** 2026-11-02 (branch `minimal/thesis-core`)

## Reason
The author uses only `live_plot_controller` and `live_plot_cascade` and asked
for a visually clearer and faster version, one window per layer, and a third
window with general information: infeed by generation type (merit order) and
the MIQP objective, "to see whether we actually improve".

Author decisions (2026-11-02, asked before implementation):
- split by layer: TS window, STS window, overview window;
- objective tile shows the plant-level cost Φ **and** the step-QP optimum J*;
- overview extras: Q reserve per zone, Q provision by source;
- discrete TS actuators drawn as raster lanes.

## Established fact that drove the objective design
`ControllerOutput.objective_value` (recorded as `zone_tso_objective` /
`dso_objective`) is the optimum of the step QP,

  J*_k = min_w  wᵀ G_w w + (∇Φ_k + 2α g_u u_k)ᵀ w + g_z ‖z‖²,

i.e. the change the local model predicts.  It is ≤ 0 whenever w = 0 is
feasible and tends to 0 at a fixed point, so it shows stationarity, not
whether Φ decreases.  Φ itself was evaluated nowhere.

## Changes

### Controllers (diagnostic only; the step is unchanged)
- `controller/base_controller.py`: `ControllerOutput.objective_terms`
  (optional, default `None`); `BaseOFOController.objective_terms(measurement)`
  (base: `None`) and `_usage_cost()` = Σ g_u u² with the MIQP's g_u vector;
  `step()` evaluates Φ right after the gradient (Step 5b), i.e. on the same
  measurement and after the AVT / qw resets.
- `controller/tso_controller.py`: `TSOController.objective_terms` — terms
  `v`, `q_pcc`, `loss`, `res_sg`, `res_der`, `u_reg`, each with the activation
  condition of the gradient term it integrates.
- `controller/dso_controller.py`: `DSOController.objective_terms` — `q_iface`,
  `v`, `u_reg`.  For `gamma_oltc_q != 1` the gradient's OLTC columns are
  attenuated, so the tap step is not the exact gradient of this Φ (stated in
  the docstring).
- `controller/central_controller.py`: own override (per-bus V weights only),
  since it inherits from `TSOController` but has a different objective.
- `tests/test_objective_terms.py` (new): finite-difference ∂Φ/∂y equals the
  TSO output gradient; for the DSO (γ = 1) the FD gradient projected through
  H equals `_compute_objective_gradient`.

### Records (all defaulted; older pickles load)
`experiments/helpers/records.py`, `MultiTSOIterationRecord`:
`zone_tso_cost_terms`, `dso_cost_terms`, `gen_mix_p_mw`,
`total_load_res_p_mw`, `zone_q_source_mvar`, `zone_q_reserve_up_mvar`,
`zone_q_reserve_dn_mvar`, `tso_shunt_step`.  Recorded every step (plant
truth) by `_record_overview_observables` in
`experiments/runners/_multi_tso_helpers.py`; generation types classified once
by `_build_gen_mix_groups` (machines by `net.gen.type`, G1/ext_grid as
"Equivalent", wind/PV split TS/STS).

Conventions (generator convention, injection into the TS):
- `STS` source = −Σ Q_PCC(EHV side) − tertiary MSC/MSR injection, i.e. the
  sub-transmission exchange including coupler reactive losses.
- `Ties` = net import over the zone's tie lines.
- `Equivalent` = machines typed `Equivalent` (G1, 10 GVA anchor), kept out of
  `SG` in sources and reserves: in zone 1 it would otherwise be ≈ 9.4 GVar of
  "SG" up-reserve and hide the zone's own ≈ 1.8 GVar.
- Reserve up/down: SG from the Milano capability at current P and V_set;
  TS-DER from the grid-code capability at current P; MSC/MSR as switchable
  nameplate steps at 1 pu; STS from the capability interval the DSOs last
  reported (NaN if none).

### Live windows (new modules)
- `visualisation/live_base.py`: growable numpy `Trace` buffers, artists
  created once and updated in place (`set_data`), y-limits from the buffers,
  wall-clock-throttled redraws (`live_plot_refresh_s`, default 2 s) and a
  forced final frame; contingency markers on every time axis; status text
  (sim time, steps/s) in the header.
- `visualisation/live_ts.py` (`TSLiveWindow`, metered view, TSO ticks):
  outputs (V per zone, tie Q, SG Q, normalised reserve, optional loading) |
  actuators (AVR setpoints, TS-DER Q, ΣQ_PCC set vs metered, machine-trafo
  tap lanes, MSC/MSR level lanes with current value and switch count).
- `visualisation/live_sts.py` (`STSLiveWindow`, metered view, DSO ticks): one
  row per DSO — interface Q (set / metered; capability edges dotted, the
  filled bands of the old plot covered the tile) | HV voltages | DER Q |
  coupler taps — plus an RMS interface-tracking-error strip.
- `visualisation/live_overview.py` (`OverviewLiveWindow`, plant view, every
  step): infeed stacked in merit order + load; Q injection by source per zone;
  up/down Q reserve per zone stacked by source (STS hatched, equivalent
  outermost and off-scale); Φ_TS and Φ_STS per controller with |J*| dotted
  (log); Φ_TS and Φ_STS shares by term.

### Removed (author: not used)
- Flags `live_plot_system`, `live_plot_tracking`, `live_plot_sbx` and the
  modules `plot_system_power_flow.py`, `plot_tracking.py`, `plot_sbx.py`;
  the old `plot_tso_controller.py` / `plot_cascade_dso.py` (replaced).
- Runner: old plotter construction and update block; the dead
  `gen_limits_static` table (only the system plot read it).
- `experiments/014_SBX_SINGLE_DEMO.py` no longer saves `sbx_h_mechanism.png`.

### Config and callers
- `configs/config.py`: new `live_plot_overview` (default False) and
  `live_plot_refresh_s` (2.0); docstrings of the remaining flags updated.
- `experiments/run_multi_system_ofo.py` `make_config`: `live_plot_overview=True`.
- Headless callers switched `live_plot_system=False` → `live_plot_overview=False`
  (tuning, tests, ch_9_3, run_case1, CIGRE 005, rms_cosim_config, capture_H);
  `tuning/parameters.FIXED_OVERRIDES` and the two plot-off lists likewise.
- `experiments/helpers/run_params._FINGERPRINT_IGNORE`: new fields added, the
  removed names kept so configs unpickled from older runs fingerprint as
  before.  Fingerprints of current configs are unchanged.

## Verification
- Tests: `tests/test_objective_terms.py` (new, 2), `test_tso_output_gradient`,
  `test_controller`, `test_oltc_cooldown`, `test_g_w_adapter`,
  `tests/tuning/test_parameters`, `test_rms_cosim`, `test_dso_voltage_band`,
  `tests/experiments` — all pass.
- Headless runs of `make_config()` + `SELECTED_PARAMSET` (90 min, all three
  windows, Agg): no errors; balance check on the last record
  Σ gen_mix − Σ res_load − losses = 0.0 MW (5005 − 4954 − 51).
- Last-record values (t = 90 min, after the G2 trip at 30 min):
  Φ_TS = 3.91 / 7.6 / 10.5 (Z1/Z2/Z3, V term only; the other TS weights are
  0 in this set), J*_TS = −0.007 / −0.076 / −0.075;
  Φ_STS(DSO_1) = 5.26 (q_iface) + 0.31 (v), J* = −0.43.
  Reserve up/down without the equivalent: Z2 1271/1316 Mvar of which STS
  353/667; Z3 731/1172 of which STS 115/233; Z1 2016/903 (+ equivalent
  9398/4049).
- Draw cost on the same 270 records (replay, Agg, half-screen size;
  overview full-screen):

  | | per redraw | redraws |
  |---|---|---|
  | old TS plotter | 425 ms (start) → 503 ms (end) | every step |
  | old DSO plotter | 542 → 688 ms | every step |
  | new TS window | 140 ms | throttled |
  | new STS window | 208 ms | throttled |
  | new overview | 295 ms | throttled |

  The old pair cost ≈ 325 s over the 270 steps, about as much as the
  simulation itself (≈ 377 s).  New: ingest 1–3 ms per step per window, and
  each window draws at most every max(`live_plot_refresh_s`, draw time /
  0.05) s, i.e. ≤ 5 % of wall time per window.
- Profiling showed most of the frame time was matplotlib measuring tick
  labels to place titles and axis labels; titles and labels are now at fixed
  positions, axes do not share x (limits set per frame), and finite bands
  are assigned as polygons instead of `FillBetweenPolyCollection.set_data`.
- Qt5Agg run (15 min, default refresh 2 s): windows open and place; 14–17
  throttled redraws of 175–240 ms per window over 45 steps, plotting ≈ 7.5 %
  of the 126 s wall time.

### Window layout (author, 2026-11-02)
TS and STS take half of the primary screen each; the overview the secondary
screen, full or half.  `live_plot_layout="dual_screen"` (default) gives the
full secondary screen; new `"dual_screen_half"` its left half.
`visualisation/style.py` gained `slot_geometry()` (the rectangle per slot,
used by `position_figure_in_slot`); the overview reads it and arranges its
tiles landscape on a full screen, portrait (one row per zone: Q sources |
Q reserve) on a half screen or a third.

## Open points
- The commitment base (`res/commitment-base-2026-10-31`, 72 commits ahead)
  does not have these changes; merging touches `multi_tso_dso.py` near the
  plotter block and the record loop.
- Φ_TS includes the usage term Σ g_u u²; when g_u acts on V_gen it adds a
  near-constant offset (V ≈ 1 pu), which the composition tile makes visible.
- `zone_tso_shunt_states` (pre-existing recorder) maps banks by bus; with
  `tso_shunt_msc_coupling_pos == tso_shunt_msr_coupling_pos` both banks would
  read the first one.  The new lanes read `tso_shunt_step` by shunt index.
