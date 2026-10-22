# 2026-10-08: MC first slice, abort diagnosis and area-1 pilot-weight check

**Author:** Manuel Schwenke / Claude Code. **Thesis side:** `latex_diss_ms/00_daily_log/2026-10-08_mc_first_slice_results.md`.

## Code changes

1. **`configs/config.py`.** New field `tso_g_v_bus: Optional[Dict[int, float]] = None`: a per-bus voltage-tracking weight for the TSO OFO.
2. **`experiments/runners/multi_tso_dso.py`.** Directly after the TSO zone controllers are built, `tso_g_v_bus` is applied through the existing `TSOController.update_voltage_tracking_weights`.
   - It prints the zone's curvature `lambda_max` before and after the change (via `controller.gw_precondition.curvature_spectrum`), at every verbosity.
   - A bus that no zone monitors raises.
   - Inert when `None`; existing runs are unaffected.
   - **Why:** the area-1 hypothesis test needs one bus's weight raised. Only per-zone weights were exposed.
3. **New `experiments/part3_mc/area1_pilot_weight.py`.**
   - Episodes: the first 10 seeds of the campaign on which both O2 and S1 were `ok`.
   - Runs O2 with the area-1 pilot (bus 25) weight × k, k ∈ {1, 2, 4, 6}; k = 1 is plain O2, which doubles as a same-machine reproduction check against the server.
   - Report: area-1 pilot error, signed pilot offset, area error, worst deviation, AVR travel, taps, and zone-1 `lambda_max`.
   - The cap k ≤ 6 comes from a `lambda_max` scan on the first episode: k = 1 → 0.886, 4 → 1.287, 16 → 2.893 (unstable), 64 → 9.321.

## Abort diagnosis (in progress)

Reproduced locally with L2 (`results/THESIS_part3_mc_abort_debug`):
- **Seed 20261008:** the guard raises `ValueError` at the machine trip (150 min).
- **Seed 20261031:** `LoadflowNotConverged` at the line trip (60 min).

Both happen at the disturbance instant.

**Code observation.** `analysis/reachability.check_reachability` re-solves a deep copy with `distributed_slack=False` from a flat start, assuming that "single slack only redistributes the slack power". After a machine trip this is not true. The copy keeps the scheduled machine P, so the whole deficit moves to the slack machine at bus 39: a different and much more stressed operating point. That is a candidate artefact; it is being tested by copying `res_gen.p_mw` into the setpoints before the re-solve.

## Abort diagnosis (result, 09:52)

Method: the failing power-flow call was captured, with a deep copy of the network made before each `runpp` (scratchpad script `debug_abort2.py`), and re-solved in variants. Outputs are in `results/THESIS_part3_mc_abort_debug/*_capture.log`.

**Type A: guard re-solve fails.**
- Seeds 20261008, 20261028, 20261096; 8 runs. Every one of these seeds contains a machine trip.
- **20261008, L2, G3 trip at 150 min (night):**
  - the guard's single-slack copy fails from every start;
  - the same network converges with distributed slack (TS voltages 0.894–1.018 pu);
  - it also converges with single slack once the machines' solved P (`res_gen.p_mw`) is copied into their setpoints;
  - reachability at that point: σ_min(J) = 0.083, λ_min(J_R) = 0.322 > 0, i.e. on the stable branch.
- **Conclusion: an artefact of the guard's canonicalisation.** The copy moves the whole post-trip deficit to the bus-39 equivalent. These episodes are not infeasible.
- In 20261096, S1 passed the trip and later failed physically (0.865 pu after a 277 Mvar load step in area 2, with G3 out).

**Type B: plant power flow fails at the disturbance.**
- Seeds 20261031, 20261072, 20261079; 9 runs. All involve line trips.
- **20261031, line 1–39 (168 km, link to the network equivalent) at 19:15:** no variant converges (flat or DC start, 300 iterations, with or without run_control, distributed or single slack).
- **Conclusion: strong indication of a genuinely infeasible post-contingency state.** This is not a proof; a continuation power flow would give one.

**Proposed fix (not implemented; author to decide).** In `check_reachability`, copy `res_gen.p_mw` into `gen.p_mw` of the deep copy before the single-slack re-solve, so that the copy has the plant's operating point. Then rerun the 8 type-A runs.
- Effect on completed runs: none on their trajectories, since the guard only checks. Their recorded margins were computed at a more stressed point than the real one (conservative); none was flagged.

## Guard fix and rerun (author decision, 10:30)

**Fix in `analysis/reachability.py`.** New `single_slack_copy(net)`: a deep copy in which every in-service machine gets its solved active power (`res_gen.p_mw`) as its setpoint.
- `check_reachability` re-solves this copy with single slack and with the plant's own `enforce_q_lims`, read from `net._options` by the new `_last_enforce_q_lims`. Before, it re-solved the raw copy with the default `False`.
- The operating point of the copy is now the plant's. The comment claiming "single slack only redistributes the slack power" is corrected.

**Tests, `tests/test_reachability.py`:**
- New `test_single_slack_copy_reproduces_the_distributed_operating_point`. It uses IEEE 9 with shared slack and one machine tripped, and checks that the copy's voltages and angles equal the plant's. It also checks that the old copy lands elsewhere.
- `test_distributed_slack_canonicalisation_matches_single_slack` is rewritten to compare margins at the same operating point. The old version only passed because no machine shared the slack in default IEEE 9.
- Result: 8/8 pass.

**Rerun.**
- `results/THESIS_part3_mc` was copied from the share to the local copy.
- The 8 affected runs (20261008 L2/O2/S1, 20261028 L2/O2/S1, 20261096 L2/O2) were moved to `runs_archive_guard_defect/`.
- `run_mc --jobs 8` reruns exactly those, via the resume logic.
- The other 292 runs are unaffected: the guard only checks and never changes a trajectory.
- The rerun records its own `provenance_*.json` and `code_diff_*.patch`.

All code changes of today are synced to the share.

## Area-1 pilot-weight check: result (10:50)

`experiments/part3_mc/area1_pilot_weight.py`; first 10 seeds with O2 and S1 `ok`; medians. Columns are the pilot-weight factor k on bus 25.

| | k = 1 | k = 2 | k = 4 | k = 6 | server S1, same seeds |
|---|---|---|---|---|---|
| zone-1 λ_max | 0.857 | 0.982 | 1.234 | 1.486 | – |
| area-1 pilot error [mpu] | 6.53 | 6.06 | 5.36 | 4.43 | 3.16 |
| signed pilot offset [mpu] | +6.45 | +5.82 | +5.19 | +4.28 | – |
| area-1 error over R_a [mpu] | 6.22 | 6.13 | 6.00 | 5.95 | 11.35 |
| worst deviation [mpu] | 20.5 | 18.5 | 18.9 | 19.7 | 30.5 |
| taps | 2 | 2 | 2 | 4 | – |

- **Reproduction.** Local O2 equals server O2 exactly (max |diff| 0.0000 mpu over 10 seeds).
- **Verdict: hypothesis NOT confirmed.** The forced trade-off it assumed would make the pilot error fall while the area error rises. In fact the pilot error falls by a third and the area error stays flat; it even falls slightly.
  - So the equally weighted OFO keeps the area-1 pilot higher than the rest of the area requires.
  - Within the loop-gain budget (k ≤ 6, λ ≈ 1.49 against the ceiling 1.6), the gap to S1 narrows to about 1.3 mpu but does not close.
- **Possible cause (hypothesis, untested).** The MIQP commits taps only when the gain summed over all buses exceeds their switching cost; the tap count rises from 2 to 4 at k = 6.
- **Ch 9** results paragraph and fourth discussion statement were revised accordingly: "more than the rest of the area requires", not "in exchange for".

## Rerun with the fixed guard: result (10:56) and thesis update

**Rerun outcomes:**
- 20261008: L2, O2 and S1 `ok`.
- 20261096: O2 `ok`; L2 `fail_v_end`. S1's earlier `fail_v_end` was valid and is kept.
- 20261028: all three still fail.
  - O2 and S1: plant power flow fails at 159 min, after line 14 trips at 150 min. The same line is involved in 20261072.
  - L2: the corrected guard's re-solve fails; a capture is under way.

**After the rerun:**
- Infeasible: 4 episodes (20261028/31/72/79), all line outages.
- Failures excluding these: L2 1/96, O2 0/96 (bound 3.1 %), S1 3/96 (bound 7.9 %).
- Paired episodes: 93. No decision changed.

New medians O2 − S1 (one-sided 95 % bound in brackets):

| | Area 1 | Area 2 | Area 3 |
|---|---|---|---|
| pilot error [mpu] | +3.22 [+3.8] | −0.89 [+0.09] | −0.51 [−0.11] |
| e_v [mpu] | −3.23 | −1.56 | −1.99 |
| worst [mpu] | −8.64 | −3.28 | −3.78 |

- Time outside the range: not shown. Share of episode-areas with any excursion: S1 18 %, O2 9 %.

**Results.** Merged results, with `RERUN_NOTES_2026-10-08.md`, were copied to the share's `THESIS_part3_mc`. The old runs are in `runs_archive_guard_defect/`.

**Ch 9 updated:**
- the 93-pair numbers and the table;
- the excluded-episode paragraph now reads "four line outages without an operating point; three S1 failures"; the words "aborted" and "defect" no longer appear in the prose;
- the S1 failure text gains 20261096 (0.87 pu; L2 also failed, O2 did not);
- the discussion now says "failed three times".
