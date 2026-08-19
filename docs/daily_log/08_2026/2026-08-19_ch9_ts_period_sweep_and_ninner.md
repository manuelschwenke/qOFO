# 2026-08-19 — Ch 9.1 Task B: the `T_TS` sweep, and what the isolated-`N_inner` probe found

**Author:** Manuel Schwenke / Claude Code
**Timestamp:** 2026-08-19, Europe/Berlin
**Reason:** Task B of the Ch 9 §9.1 handoff
(`00_daily_log/2026-08-19_handoff_ch9_experiments_A_B.md`). Eq. (9.2) fixes
`T_TS = N_inner · T_STS` with `N_inner = 9` an educated guess that has never
been evaluated. B2 (the `T_TS` sweep) is the closed-loop selection evidence;
B1 (the isolated STS) is the measurement of eq. (9.2) itself.

## What was built

| file | role |
|---|---|
| `experiments/ch_9_parameter_selection/_ch9_selected_design.py` | rebuilds the §9.3 selected weights through the campaign's own recipe and **asserts** them against the archived evaluation; refuses to run on mismatch |
| `experiments/ch_9_parameter_selection/ch_9_1_ts_period_sweep.py` | B2 — sweeps `tso_period_s ∈ {60,120,180,240,300}` at fixed `T_STS = 20 s` |
| `experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py` | B1 — isolated STS, frozen-OFO parent, CAIR-edge step |
| `configs/config.py`, `experiments/runners/multi_tso_dso.py` | `q_pcc_injection_with_ofo_parent` + `q_pcc_setpoint_schedule_per_dso`, default-off |

The weight reproduction verifies bit-for-bit: all five designed weights and
`dso_g_v = 150000.0` match `tier1_aa4f6d4a8654.json`, and the current Stage-0
fingerprint (`bca4dedc80a9`) still matches the cached design, so nothing is
regenerated.

## Three places the brief was wrong, all found by running it

1. **The coupler OLTC cooldown is not 60 s.** At the selected design
   `oltc_cooldown_s = 30.0` (wall clock), `oltc_cooldown_s_mt = 180.0`, and
   `int_cooldown = 6` **iterations** = 120 s. The binding lockout is the
   iteration count, so at `T_TS = 60 s` the changer is nominally unavailable
   for *twice* the interval, not for it. B.6's concern is real but understated
   and driven by a different mechanism than named.

2. **`dataclasses.replace(spec, tso_period_s=X)` is not "nothing else".** The
   selected design runs `coordination_mode = "sbx_h"`, and `SBXConfig` carries
   its own `tso_period_s` that the runner requires to match
   (`multi_tso_dso.py:1280`). Six of nine pilot runs raised on this. The guard
   is correct, and what it guards is a genuine coupling: `k_sched = 2` is a
   cycle length in **TSO iterations**, so sweeping `T_TS` also scales the SBX-H
   settlement cycle in wall clock (2 min at 60 s → 10 min at 300 s).
   **This is a confound in the sweep and is reported, not hidden.**
   `--sbx-cycle` offers both readings; the default holds `k_sched`, i.e. holds
   the controller's own configuration fixed and lets the wall-clock consequence
   follow the period — which is what changing `T_TS` in service would do.

3. **`rural_700` is not a different network.** Same IEEE 39 + four 110 kV
   sub-networks as `base_410` (both `apply_wind_replace`), differing only in
   installed DER capacity per DSO: 460 MW wind + 240 MW PV against 270 + 140
   (`network/ieee39/constants.py:223`). Not a finding. But `ScenarioSpec`
   defaults to `base_410`, so the thesis should state which capacity scenario
   Ch 8's benchmark table lists.

## B2 result — 60/60 runs, exit 0

Pooled over 12 design windows, ρ_k = residual the next dispatch inherits,
normalised by the correction commanded.

| `T_TS` [s] | `N_inner` cfg | ρ median | ρ p95 | censoring | lockout occ |
|--:|--:|--:|--:|--:|--:|
| 60 | 3 | 0.662 | 4.13 | 0.21 | 0.02 |
| 120 | 6 | 0.375 | 2.95 | 0.19 | 0.02 |
| 180 | 9 | 0.308 | 3.10 | 0.19 | 0.02 |
| 240 | 12 | 0.273 | 3.20 | 0.20 | 0.02 |
| 300 | 15 | 0.215 | 2.97 | 0.19 | 0.02 |

**The censoring fraction is flat at ~0.19–0.21 across every period.** Giving
the subordinate layer 15 iterations instead of 3 does not reduce the fraction
of dispatch intervals in which the tracking error never enters the band and
stays. Whatever fails to settle, fails for a reason that is not the number of
iterations available. The median residual does improve monotonically, but the
p95 plateaus after 120 s.

**Lockout occupancy is ~0.02 everywhere**, so B.6's second risk — that
`T_TS = 60 s` is confounded because the changer is locked out for the whole
interval — is **not** supported. Taps barely move at all (0.009 moves per
interval at 60 s). Per unit time the tap rate is roughly constant across the
sweep, so the period choice carries no wear penalty.

### No U-shape in ρ, and ρ cannot show one

ρ decreases monotonically. That is **not** evidence against the expected
U-shape: ρ measures how well the subordinate layer executed the correction it
was *told* to make, not whether that correction was still right when it
landed. The stale-setpoint cost lives in the supervisory tracking objective
(`f_ts`, `f_q`), which this script does not compute. Finding the U-shape needs
a supervisory-level metric — **an open item, and the selection argument for
180 s is not yet measured.**

## Two defects in my own reporting, corrected

- **The voltage band was hardcoded 0.95/1.05** while the selected design
  carries `v_min_pu`/`v_max_pu` = **0.9/1.1**. Against the band the study does
  not impose, 37–42 % of intervals looked like violations; against the
  configured one it is **0.7 % at 60 s and 0.000 at every longer period**. The
  limits are now read from the config.
- **`n_k` was reported as a pooled p95**, which with ~20 % censoring lands
  inside the censored region — every censored interval reports `n_k` = the
  interval length, so the p95 *was* the cap at all five periods and measured
  the horizon rather than the loop. Uncensored quantiles are now reported
  alongside the censoring fraction. Over uncensored intervals the median is
  **0 at every period**: when the loop settles, it settles immediately.

## B1 — the probe found the experiment measuring nothing

The first probe returned `N_inner = 0` for every transformer and both
directions. It was not a fast loop: `band_w = 0.00` and `step_mvar = 0.00`.
The reported capability had collapsed to `cap_min == cap_max == q_now`, so the
band-edge target equalled the current flow and no step was applied.

**The collapse is not caused by the frozen parent** — it reproduces with a
live 180 s supervisory layer — and it is not rare: the sweep's own per-interval
widths have median 67–181 Mvar but **minimum 0.00 for all twelve interface
transformers**. Reading a single instant lands on a degenerate band often
enough to matter.

Fixed: the band is taken from the last sample at or before the step instant
whose width clears a floor, the operating point stays at the step instant, and
a transformer with no usable band — or a step below the settling band — is
returned as an explicit failed case with a reason, never as a zero-step
measurement.

**Open question for the author:** why the reported CAIR collapses to zero width
at some operating points at all. It is a different symptom from the known
voltage-blindness of the capability message
(`2026-08-18_dso4_voltage_relief.md`), and it means the supervisory layer
periodically sees "nothing available" at interfaces that in fact have ~100 Mvar
of range.

## Status

- B2: complete, 60/60, re-running at the corrected commit for a clean run of
  record.
- B1: plumbing fixed and self-tested; probe re-running on DSO_1 and DSO_4.
- Task A: **failed** — see
  `2026-08-19_ch9_settling_table_emitter_rework.md`; `ComInc (RMS init) failed`
  after ~30 min of non-convergence in the preflight. Not a code defect in the
  battery; the RMS model does not initialise in its current state.

## Unresolved

- The supervisory-level staleness metric that would show the U-shape.
- `N_inner` itself: B2 says the in-situ answer is 0 when it settles at all,
  with a ~20 % structural non-settling floor that `N_inner` does not move.
  Whether that supports keeping `T_TS = 180 s` is an author decision.
- The circularity stands as stated: `G_w` was calibrated with `N_inner = 9`
  assumed, so all of this is a check on the guess, not an independent
  measurement.
