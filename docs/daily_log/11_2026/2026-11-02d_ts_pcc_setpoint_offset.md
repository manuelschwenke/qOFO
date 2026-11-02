# 2026-11-02 — TS live window: Q_PCC setpoint includes the shunt feedforward

## Problem (author, 2026-11-02)
The "Q_PCC to STS" tile of the TS live window plotted Σ u_PCC per zone, the
TSO MIQP decision.  The STS is dispatched u_PCC + q_itf_sh_offset: after a
committed MSC/MSR level change the runner adds the switched-shunt
feedforward (from the hosting DSO's own 3W model) to every coupler setpoint
of that STS, and the cached models are corrected by the rank-1 update.  The
metered interface flow tracks the shifted value, so after the MSC switch at
t ≈ 120 min the tile showed a gap of order 60 Mvar (zone 2) although the
tracking was good.

## Change
- `experiments/helpers/records.py`: new field
  `zone_q_pcc_sh_offset: Dict[int, NDArray]`, parallel to `zone_q_pcc_set`.
- `experiments/runners/multi_tso_dso.py`: on each TSO tick, immediately
  before the setpoint dispatch (after this step's shunt commits), record
  `q_itf_sh_offset` per `pcc_trafo_indices` of each zone.
- `visualisation/live_ts.py`: the tile's setpoint trace is now
  Σ (u_PCC + offset), i.e. the setpoint the STS actually receives.

`zone_q_pcc_set` itself is unchanged (it remains the MIQP decision variable
and is used by `ch_9_3_compare_runs.py` and `018_SBXV_E1.py`).

## Verification
Headless 6 h run, selected paramset, `make_config()` scenario: three MSCs
in zone 2 switch 0→1 at t ≈ 120 min and 1→0 at t ≈ 330 min.  Zone-2 offset
non-zero on 81 of 121 TSO ticks, range [−77.6, +15.7] Mvar; zones 1 and 3
zero.  In the redrawn tile the setpoint lies within a few Mvar of the
metered flow throughout; before the change the gap was the offset itself.

## Open points
- The offset does not return to zero when the banks return to level 0:
  the switch-on steps summed to −77.6 Mvar over the three couplers of each
  STS, the switch-off steps to +93.3 Mvar (each evaluated at the then
  current operating point and tertiary voltage), leaving +15.7 Mvar of
  feedforward with all banks at 0.  This is a property of the runner's
  accumulation, not of the plot; not changed here.
- Analyses that compare `zone_q_pcc_set` with metered interface flow must add
  `zone_q_pcc_sh_offset` as well; `dso_trafo_q_set_mvar` already holds the
  shifted value on the STS side.
