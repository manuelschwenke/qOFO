# Plan: Experiments D1/D2 (dead band of the DAD layer) on the QSS plant

**Status:** PROPOSAL 2026-10-27, for discussion. Nothing implemented.

**Basis:**
- `Chapters/Chapter09.tex` §`ch:timescales:deadband`: equations
  `eq:param:deadband:drift` and `eq:param:deadband:detector`, and the two
  `\todo`s.
- `Chapter08.tex` §`ch:setup:framework:modes`: the plant is solved between
  dispatch instants with the anchors held.
- Plan B of `qOFO_GH/docs/ch9_F1F2_qss_and_ch11_campaigns.md` (2026-09-20).

## 1. What the thesis requires

| | D1: routine droop inactivity | D2: disturbance response |
|---|---|---|
| Runs | W1–W6, 2 h uninterrupted, droop suppressed; then confirmation runs at the selected width Δ\* and at the adjacent narrower one | W1–W6 × 3 disturbances (+200 Mvar at TS bus 7, ±200 Mvar at bus 11) × 3 legs (Δ\*, a wider width, droop disabled) = 54 cells |
| Plant | QSS, solved at dispatch instants **and** at every profile step and event instant between them, DAD anchors held | as D1; disturbance applied **between** two dispatch instants |
| Recorded per converter and solution instant | held anchor, U_f (voltage entering the dead zone), droop increment (unsaturated, saturated), delivered Q | as D1, plus verification that every event reached the plant |
| Result | d_ik = max over [t_k, t_k+1) of \|U_f − U_anchor\|; Δ\* = min{d ∈ 𝒟 : d ≥ max d_ik + ε_U}; activation counts at Δ\* and the narrower width | settled incremental support, voltage deviation left until the next STS dispatch, voltage extrema, hand-over to OFO (2 representative cases) |

Fixed throughout: the final weights, T_TS = 180 s, T_STS = 20 s, the slope.
There is no measurement noise; the thesis states this.

## 2. What the code already does (verified 2026-10-27)

1. **Profile steps are already solved.** Profiles are 15-min data,
   linearly interpolated to `dt_s` (`core.profiles.load_profiles`). At
   dt = 20 s the plant (`PandapowerStaticPlant`, `runpp(run_control=True)`,
   droop settled by `QVLocalLoop`) is solved at every step. "Every profile
   step" is therefore every dt.
2. **Anchors are held between dispatches.** `apply_tso_controls` re-anchors
   only the TS DERs, at TS dispatch; `apply_dso_controls` re-anchors only
   the STS DERs, at STS dispatch. Between two TS dispatches the TS-DER
   anchors are held over 8 intermediate solves. At dt = T_STS = 20 s the
   STS DERs have **no** solve strictly between their dispatches: the
   pre-dispatch solve at t_k+1 is the only held-anchor point.
3. **Events fire only on the dt grid.** They match `|t_event − t| < 1e-9`
   and are applied after the profile solve and before dispatch. At
   dt = 20 s every step is an STS dispatch, so the STS reacts in the same
   step and D2 has no "until the next subordinate dispatch" window.
4. **A `q_step` on a profiled load lasts one step.** `apply_profiles`
   rewrites `q_mvar = base_q_mvar × profile` at every step, so the step is
   overwritten. This is the failure mode of the archived RMS battery
   ("identical exports across events"). A **dormant-load `connect`** event
   (unprofiled, `p = 0`, `q = ±200 Mvar`) persists.
5. **No per-converter records** (records.py holds totals only).
6. **The droop is modelled consistently.**
   - Dead band and slope come from `tso_/dso_qv_deadband_pu` (default
     0.01) and `tso_/dso_qv_slope_pu` (0.06); a common half-width means
     setting both.
   - The OFO uses the bare H (`apply_qv_h_transform = False`), so
     suppressing the droop (dead band 1 pu) does not create a model
     mismatch.
7. **The runner files are being changed elsewhere.** The parallel branch
   `case2/ref` (6 commits on 04b0d42) modifies `multi_tso_dso.py`
   (+309 lines), `records.py`, `plant_io.py`, `der_qv_local_loop.py` and
   `config.py`.

## 3. Proposal: no edit to the runner

- **Time grid: dt = 10 s** (T_STS 20 s, T_TS 180 s).
  - This gives one held-anchor solve between STS dispatches and 17
    between TS dispatches.
  - Events at odd multiples of 10 s fall strictly between STS dispatches.
  - The runner already supports dt < T_STS (`_is_period_hit`).
  - Check before use: W1, 2 h, dt 10 vs 20. Voltages and decisions at the
    dispatch instants must agree within PF tolerance. Exact identity is
    not expected, because the warm starts differ.
- **Recording: a probe, not a runner change.**
  - A pandapower controller `DroopProbe` is inserted through
    `pre_loop_hook`. Its `finalize_control` runs after every converged
    control loop (pandapower 3.4.0, verified).
  - At each solve it stores, per `q_mode == "qv"` DER: time, level
    (TS/STS), anchor, U_f, q_set, unsaturated and saturated droop
    increment, delivered Q and a dead-zone flag.
  - The time stamp comes from a `PandapowerStaticPlant` subclass passed
    as `plant_factory`; it records `t` in `apply_exogenous`.
  - Each solve is classed pre- or post-dispatch by whether its anchor
    changed.
  - This leaves zero blast radius on other experiments and no conflict
    with `case2/ref`.
- **Events:** dormant-load `connect` events at TS bus 7 and 11, at an
  off-STS-grid instant. Each is verified from the records (bus Q injection
  before and after).
- **Producers** in `experiments/ch_09_parameter_selection/`:
  `_droop_probe.py`, `ch_9_4_d1_droop_inactivity.py`,
  `ch_9_4_d2_disturbance.py`. They write documented runs through
  `start_thesis_run` (target `ch:timescales:deadband`; feeds the D1/D2
  result paragraphs).
- **Compute** (≤ 30 workers):
  - D1: 6 suppressed + 12 confirmation runs × 2 h (720 steps), < 1 h.
  - D2: 54 runs × about 1.5 h (warm-up, event, 30 min observation), about
    1 h.

## 4. Decisions for the author

1. **Weights:** D1/D2 run at the procedure's result
   `ch9_selected_band_2026_10`. This needs `SELECTED_PARAMSET` switched
   first. The margin check supports it: ≥ 0.018 pu to 1.10 pu; the band is
   exceeded once, by ≤ 0.0023 pu (DSO_4, W1).
2. **Time grid:** dt = 10 s (proposed), or 5 s (4× the solves, no new
   information from linearly interpolated profiles).
3. **Candidate set and margin.**
   - Proposed 𝒟 = {0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020} pu; 0.010
     is the current default and the earlier comparison value.
   - Proposed ε_U = 0.001 pu: converter voltage-measurement accuracy, far
     above the PF tolerance.
4. **"Wider width" in D2:** the next candidate above Δ\* (proposed), or 2Δ\*.
5. **Disturbances.**
   - Sign convention: "+200 Mvar" read as additional reactive **load**
     (absorption); bus 11 both signs.
   - Event time: window start + 60 min + 10 s (off the STS grid, inside a
     TS period). Observation: 30 min (10 TS periods).
   - Warm-up rule for D1: the first 30 min reported separately.
6. **Recording:** the probe (proposed), or record fields in `records.py`
   plus a runner change after `case2/ref` is merged.

## 5. Risks / open points

- **dt changes the closed loop slightly.** dt 10 s is not bit-identical
  to the dt 20 s runs of the weight selection. The check in §3 bounds
  this.
- **The no-crossing selection holds for these windows only.** The thesis
  already says so.
- **`case2/ref` merge:** it carries an import of the old
  `experiments.ch_9_parameter_selection` path (`run_case1.py:78`), which
  must be updated on merge.
