# 2026-11-06: overnight runs, scenario F's second step did not fire, fix and reruns

Continues `2026-11-01b_ch9_settling_failure_diagnosis.md` (entries up to 2026-11-05 22:42).

## 07:17: status of the overnight runs

All four runs completed, and every variant reports "ok" with 1080 records (360 min):

| run | host | finished | variants |
|---|---|---|---|
| REF E, fixed machine taps, `ref/0019_2026-11-05_215935_run_ref_v4_E_fixmt` | e5wimi | metrics.json written | L2, S1, S2, O2, O3, M |
| Case 1 on A, `ch_10_case1_svr/0001_2026-11-05_215956_run_case1_v4_A` | e5wimi | 03:18 | L2, S1m, S1, O0, O1, O2 |
| REF F, `ref/0020_2026-11-05_223831_run_ref_v4_F` | e5power | 02:22 (pulled) | all six |
| REF FM, `ref/0021_2026-11-05_224139_run_ref_v4_FM` | e5power | 02:21 (pulled) | all six |

The Case 1 wall times are 42–62 min for L2/O1/O2 and 4.7–5.3 h for S1m/O0/S1.

## 07:20: F's second step never fired

- **Evidence.** Windowed metrics (shared notes `sts_access_gain_scripts/F_windows.py/.out`):
  - After 271:10, O2 restores area 2 by itself (signed level +1.1 mpu over the last 30 min, min
    1.003 pu), against a TS-a2 bound of −89.6 mpu. LOC's level hardly moves at 271:10 (−18.9 →
    −19.8 mpu).
  - `events_fired` lists three events in every variant (1840, 7300, 12740 s). The area-2 load Q
    stays at 558–559 Mvar across 271:10.
- **Cause.** The runner fires an event only at a step whose time equals the event time
  (`experiments/runners/multi_tso_dso.py:3877`, tolerance 1e-9 s). The step is 20 s, and step 2
  was at 271:10 = 16270 s, which is off the grid. My error: the offset of 70 s was chosen for a
  distinct phase in the 180-s period without checking the 20-s grid.
  - Last night's runner check showed the dormant loads and the event list, but not whether each
    event would fire. `test_scenario_F` checked distinct phases, not the grid.
- **Consequence.** 0020/0021 are E's schedule with a single 400 / 350 Mvar step at IEEE 4. They
  carry a `NOTE_step2_not_fired.md` and are not reported as F/FM.
- **What they show (single step, not scarce by the bound +12 mpu).** O3 − O2 in area-2 e_v after
  212:20 is +0.1 mpu. This is consistent with F's hypothesis 1, which the rerun reproduces
  because its first 271 min are identical.

## 07:22–07:25: fix and declaration amendment

- **`ref/scenarios.py`:**
  - F and FM step 2 at `_at(270, 60)` = 271:00 (16260 s, phase 60 s; was 271:10).
  - `DT_S = 20.0`; `_at` raises for a time off the step grid.
- **`tests/test_scenario_F.py`:**
  - the new step-2 time;
  - every event of every scenario lies on the grid;
  - `_at(270, 70)` is rejected;
  - result: 3 pass, plus `test_ref_remote_driver` (1).
- **MC campaign:** not affected. `PHASE_GRID_S` = 20–160 s in 20-s steps; the 180-s prescreen runs
  without events.
- **Declaration:** amendment of 2026-11-06 in `docs/analysis/2026-11-05_scenario_F_declaration.md`
  (what happened, what was seen, the change). Hypotheses 2–5 are unchanged.

## First look at REF E with fixed machine taps (run 0019; not yet evaluated for Ch 10)

Area 2, after the load step (212:20–360):

| | L2 | S1 | S2 | O2 | O3 | M |
|---|---|---|---|---|---|---|
| e_v (mpu) | 37.2 | 37.1 | 21.1 | 21.8 | 18.0 | 14.4 |
| min V (pu) | 0.952 | 0.868 | 0.927 | 0.939 | 0.946 | 0.974 |

- S2 − S1 = −16.0 mpu and O3 − O2 = −3.8 mpu.
- S1 reaches 0.868 pu and stays below 0.95 pu for 3.8 % of the window. This is to be examined in
  the Ch 10 evaluation, together with the tap analysis.

## 07:25–07:30: F and FM relaunched on e5power (commit fb0010c)

- F: tag v4_F_r2, e5power run 0022 (driver started 07:25:02, remote launch 07:26:30).
- FM: tag v4_FM_r2, e5power run 0023 (driver started 07:26:42, remote launch 07:30:03).
- Code copied over e5power's 3d98d31 (28 files, hashes equal; includes the new `scenarios.py`).
- 6 jobs each. Last night's pair took 3 h 44 min, so both should be back around 11:10–11:20.
- Driver logs: `results/ch_11_case2_multisystem/ref/_remote/v4_F_r2|v4_FM_r2/driver.log`.

## 08:45: machine-tap analysis on runs 0019 and 0001 (author: "Yes, both"; output written 08:45:30)

Question (author 2026-11-05): "Keep fixed. Analyse afterwards if a tap change would have been
needed." Script and output: shared notes `sts_access_gain_scripts/tap_analysis.py/.out` (logs
only, no new simulation).

- **Relay rule in the code** (`controller/local_rules.TapRelay`, as installed in
  `multi_tso_dso.py`): one tap step when the HV side leaves 1.03 ± 0.01 pu, at most one per
  900 s; the slack machine (G1) has none.
- **Criterion "needed".** HV side outside the band while the machine itself cannot correct it:
  - HV low and (AVR at 1.05 pu or q_max − q < 2 % of the range), or
  - HV high and (AVR at 0.95 pu or q − q_min < 2 %).
  Only then would a tap add capability. Elsewhere the deviation is the controller's choice.
- **Result, REF E run 0019 (L2, S1, S2):** 0.0 % of the time for every machine.
- **Result, Case 1 run 0001 (L2, S1m, S1, O0, O1):** 0.0 %, except G3 under S1m/S1/O0 with
  1.3–1.9 % (area 2: HV side below 1.02 pu while its AVR sits at 1.05 pu or its Q near q_max;
  scenario A does not trip G3).
- **What a relay would have done.** An open-loop replay of the rule moves G10's tap 24 times
  upward under S1 and S2.
  - G10's HV side is below 1.02 pu for 99.6 % (S1) and 98.4 % (S2) of the run, min 0.957 pu.
  - Meanwhile its AVR stays at 0.984–1.019 pu (never at a limit) with ≥ 210 Mvar of margin.
  - The low voltage is the SVR's choice (pilot node plus alignment of the reactive levels).
  - In run 0017 the relays did act (53 local tap moves under S1), and G10's AVR sat at 0.950 pu
    for 87 % of the run: the wind-up that `fix_machine_taps` removes.
- **Why SVR's area-1 error rose (7.8 → 14.2 mpu).**
  - Level −4.7 → −8.2 mpu, shape 5.7 → 11.0 mpu.
  - The relays had held the machines' HV buses near 1.03 pu, a second, local voltage control.
    Without them, SVR's own objective shows: the pilot is still held (e_vp area 1: 4.0 → 4.1 mpu),
    but not the rest of area 1.
- **Consistency with the thesis.** Ch 7 and `tab:case3:variants` already state "held" for LOC,
  SVR and SVR+. Run 0017 did not match this; run 0019 does.

## 08:47–08:56: Ch 10 reference run moved to run 0019 (thesis clone, not committed)

- **Figures.** `figures_ref.py 0019 --thesis Z:\latex_diss_clone --event 3`, then
  `make_tikz_tables.py`.
  - The data folder `graphics/Ch10/data/2026-11-05_case3_ref/` is overwritten. Its README now
    names run 0019 and code 48d6193 (HEAD at launch; the manifest has no git state).
  - The 0017 data and the old `.tex` are backed up in the scratchpad (`thesis_backup_0017/`).
- **Axis limits widened** to the new ranges (scratchpad `ch10_fig_axes_0019.py`): vtrack panels
  52/142/62/94, summary 94, event 142, stsv 152, tie flows 1-2 [−125, 145] and 2-3 [−312, 88].
- **Text and table** (scratchpad `ch10_ref_E_0019.py`, each replacement asserted once):
  - provenance block of 0019;
  - table rows LOC/SVR/SVR+;
  - 35 text numbers in the voltage, interface, STS-voltage and tie-flow subsections;
  - one sentence on G10 under the classical scheme;
  - one sentence on the held taps (the tap analysis), with comments.
- **Scarcity bound in the setup text:** 27 mpu / 13 % → 29 mpu / 17 %. This is the bound at the
  run's own start point (360-min commitment), from the F declaration.
- **Correction found by the table check:** OFO's f_V^STS is 7.355 mpu (RMS over the four STS, as
  the caption says).
  - The table had 7.3; now 7.4.
  - "pay 0.7 mpu more than under OFO" is now 0.6.
- **Key changes for the reader.**
  - Law change SVR → OFO: −45 % (was −25 %); SVR+ → OFO+: −35 % (was −16 %).
  - Access under the classical law: −27 % (was −24 %), area 2 24.8 → 14.9 mpu.
  - Under OFO: unchanged, −14 %.
  - SVR's peak after the step: 138 mpu. SVR reaches 0.868 pu 20 s after the step.

## 09:05–09:12: CEN oscillation in fig:case3:tieflow (author: "what happens with CEN in figure 10.7")

Read from the logs of run 0019 (CEN is identical in run 0017):
- **What.** Between the opening of tie 9–39 (121:40) and the load step (212:20), every
  continuous CEN set-point reverses direction at nearly every 20-s iteration (sign-flip share
  0.96–1.00). This is a period-2 cycle at constant amplitude.
  - Summed TS-IBG movement: about 55 Mvar per step. Summed AVR movement: about 0.0016 pu per step.
  - All actuators flip in phase, together with the mean voltage of all three areas (area 2:
    about 9.6 mpu per step). It is one common voltage-level mode.
  - Taps, banks and coupler taps take no part. The cycle starts at 121:40 and has died out by
    about 215 min.
- **Share of CEN's effort.** 77 % of its AVR travel (0.42 of 0.55 pu) and 87 % of its TS-IBG
  movement fall in this window. The Ch 10 sentence "CEN moves the AVR set-points 18 times as far
  as OFO+" therefore mostly measures this cycle (not yet changed in the text).
- **Margin by design.** CEN's loop gain at the selected weights is 1.781 at t = 0. The runner
  scales all CEN movement weights by κ = 1.268 to reach the target λ_max(M) = 1.405
  (`M/console.log`), a factor 1.42 to the limit 2. It is evaluated at the t = 0 operating point
  with the intact topology.
- **Hypothesis (not verified).** With 9–39 open, the plant's sensitivity of this mode exceeds
  the cached one by more than that factor, so the effective per-iteration gain reaches 2.
  - The amplitude is bounded by a nonlinearity (constraints).
  - After the load step, the operating point and active set change and the cycle ends.
  - Why the cascade (same weights, 180-s TS dispatch) does not show it is not established.
- **Check proposed (not run):** the eigenvalues of the error map I − H_true·M_model at ~150 min,
  with the post-trip sensitivity.

## 09:14–09:22: Ch 9 on Case 1 run 0001 (author: "Yes, update Ch 9")

- **Front rematched to v4** (commit 6cb0d19). The K1-PAR rule puts one S1 tuning at λ_max(B)
  of the selected weights: v4 gives 1.405, so S1_T128 (T_RVR = 128.1 s) replaces S1_T137
  (1.31). The ceiling 1.6 (112.5 s) and 600 s are unchanged.
- **Front launched** 09:14:34 into run 0001: `run_case1 --only K1-PAR --scenario A --jobs 7
  --run-dir …0001…`, local, pythonw, console in the scratchpad (`case1A_par_console.log`).
  S1 runs took 4.7–5.3 h in K1-REF, so expect about 14:00–14:30.
- **Tool fixes** (this commit):
  - `report_case1.py` took the event times from the variant's make_config contingencies (the
    six old minutes) and would have measured eq:case3:epost at the wrong times for a scenario
    run. New `event_minutes()` reads the run's recorded scenario.
  - `figures_case1.py`: the same for the event lines; panel (b) now shows the disturbance and
    area with the largest growth under S1^- (on A: the G4 outage, area 1; on the six old
    events: the tie trip at 210 min, as before).
- **Results of run 0001** (`report_case1.py`, 09:17):
  - S1m = O0 to 4.7e-10 pu (bus voltages) and 1.3e-10 pu (AVR); no output constraint active.
  - Monitored voltages 0.942–1.055 pu. No regional level at its bound; AVR references at their
    limits in 2 / 90 / 99.7 % of the plant steps (areas 1/2/3).
  - Growth under S1m/O0: 3.6 (G4 outage) / 2.8 (load) / 1.1 (tie trip) mpu; under O1/O2
    ≤ 0.13 mpu. The safeguard (S1) suppresses it after G4 and the load step, but not after the
    tie trip (1.10 mpu, as S1m's 1.11). Why is open.
  - Deployed tunings, RMS over areas:

    | variant | AVR travel (pu) | pilot (mpu) | e_v (mpu) | worst (mpu) |
    |---|---|---|---|---|
    | S1 | 0.12 | 2.5 | 12.9 | 86 |
    | S1m/O0 | 0.45 | 2.9 | 13.0 | 88 |
    | O1 | 0.049 | 5.7 | 10.2 | 83 |
    | O2 | 0.052 | 5.4 | 7.2 | 83 |
    | L2 | 0 | 12.3 | 17.0 | 91 |

- **Thesis** (scratchpad `ch9_ref_A_0001.py`, 09:21; CRLF kept; backups in
  `thesis_backup_0017/`):
  - instrument paragraph (scenario A's three events);
  - the three observations of ch:case_svr:equivalence;
  - Fig. 9.1 regenerated;
  - tab:case_svr:reference (5 rows);
  - two discussion statements;
  - a todo marking the front statements and fig:case_svr:pareto as still from 2026-10-06.
  - tab:case_svr:stats stays from the old ensemble (MC campaign).

## 09:25–09:42: CEN loop gain under model mismatch (author: "yes, check it. and we should have a reference CEN that does not oscillate, so maybe we have to lower the lambda a little more?")

Script and output: shared notes `sts_access_gain_scripts/cen_mismatch.py/.out/.pkl`.
- **Method.**
  - Capture M's controller at initialisation (the runner's `debug_central_curvature` hook,
    patched to raise) and apply M's own κ (1.268).
  - Rebuild its H on the plant copy after E's disturbances (warm-started PF, plant settings) and
    take the eigenvalues of K = G_w⁻¹ H_mᵀ D_v H_t on the continuous columns (TS-IBG, AVR,
    STS-IBG).
  - A real eigenvalue > 2 means the per-tick error map flips sign and grows (period 2).
  - Operating point stays t = 0; only topology and the load step change.
  - G3's machine transformer stays energised at no load: the sensitivity builder drops a column
    when it is out of service.
- **Result** (max Re μ; in brackets without IBG1, the area-2 TS-IBG):

  | state | max Re μ | without IBG1 | cycle in the run |
  |---|---|---|---|
  | intact (model) | 1.405 | 1.12 | no |
  | 9–39 open only | 1.67 | 1.38 | — |
  | G3 out | 2.17 | 1.82 | no (30:40–121:40) |
  | G3 out + 9–39 open | 3.12 | 2.71 | yes (121:40–212:20) |
  | + 600 Mvar | 3.29 | 2.87 | no (after ~215 min) |

- **The dominant mode is carried by the TS-IBGs** (participation IBG3 1.00, IBG1 0.6–0.9, IBG2
  0.4, IBG0 0.3); no AVR is among the top participants. So the IBG movement weights, not the AVR
  weights, are the lever.
- **Confirmed:** the disturbances raise the mismatch loop gain far beyond 2 (×2.2 after G3 and
  9–39).
- **Not explained by this static, unconstrained check:**
  - Why no cycle after the G3 outage alone (2.17, marginal).
  - Why it ends after the load step (3.29). The area-2 IBG is then at its Q maximum for the whole
    window (reserve 0.000, 205 Mvar), but without it the gain is still 2.87.
  - The operating point (t = 0 here) and the active constraints decide where the cycle appears.
- **Required CEN target** (at this operating point; the largest eigenvalue scales with
  1/target): max Re μ ≤ 2 after G3 + 9–39 needs λ_target ≤ 0.90; ≤ 1.6 needs 0.72; ≤ 1.4 needs
  0.63.
- **Side findings:**
  - The `variants.py` comment says CEN's loop gain is 0.98, below the target; with v4 it is
    1.781, and κ = 1.268 is applied.
  - Ch 10 says "CEN runs at the selected weights": it runs at the selected weights × 1.268.
  - The author has seen the cascade oscillate in other windows too: the same mismatch question
    applies to the per-area TS-OFOs. Not examined here.
- **CEN wall time** in run 0019: 1.8 h (e5wimi).

## 09:44–09:50: CEN ceiling 0.90 (author: "Use 0.90 instead"), test run launched

- **Code** (commit aea9def):
  - `variants.M_LAMBDA_TARGET = 0.90` replaces the selected design's λ_max_B (1.405) as M's
    ceiling over the continuous actuators.
  - The stale comments on M's loop gain ("0.98") are corrected.
  - Test `test_m_targets_the_declared_loop_gain_below_the_selected_designs`.
  - 13 of 14 Case 2 variant tests pass. The failing one, `test_m_is_ready_and_has_the_cascade_timescale`,
    expects M's tap/bank lock at 18 min (n = 53). Since the device-limit harmonisation (c744d89)
    the code uses `tso_int_cooldown` (15 min, n = 44). Stale test, unrelated, not changed.
- **Static check at 0.90:** the largest eigenvalue scales to 3.117 × 0.90/1.405 = 2.00 after
  G3 + 9–39, and to 2.11 after the load step (there no cycle was seen even at 3.29). The margin
  is thin; the run decides.
- **Run-number reservation:** e5power holds 0022/0023 (F/FM reruns, not yet pulled). The empty
  local folder `ref/0023_2026-11-06_072958_run_ref_reserved_for_e5power_0022_0023` keeps a local
  run from taking those numbers. Delete it after the pull.
- **Test run:** REF E, M only, tag v4_E_M090 → `ref/0024_2026-11-06_094814_run_ref_v4_E_M090`,
  launched 09:48:00 (local, pythonw).
  - Console: `[central] g_w x kappa = 1.979 (target 0.9): lambda_max 1.781 -> 0.9`.
  - At the 1.8 h of run 0019, expected around 11:40.
  - If the cycle is gone, M of run 0019 is replaced, and the Ch 10 CEN numbers and the sentence
    "CEN runs at the selected weights" (now × 1.979) are corrected.

## 10:05–10:45: scenario E2 defined, declared and launched (author: "then do such a scenario as i described, with 9-39 in operation, and maybe G3 and load step overlapping, then G3 coming back, then a line drop later"; "I will be gone now for an hour. Please work autonomously")

- **Plant-only candidates** (shared notes `candE.py/.out`, `candE_stage1.out`).
  - First pass: the TS-a2 bound left G3's AVR at 1.03 pu in states with G3 back. Corrected:
    G3's AVR at 1.05 pu counts as area 2's own resource. Output of the biased pass kept as
    `candE_v1_G3avr_fixed.out`.
  - The scarcity comes from G3 plus the load (TS-a2 −28.3 mpu without 9–39, −26.6 with it).
    The 9–39 opening mainly raises the mismatch.
- **Mismatch loop gain for all candidate states** (`cand_mismatch.py/.out`): CEN at 0.90, and
  the TS-OFO of each area under O2/O3.
  - Every E2 state is < 2: CEN ≤ 1.45, cascade ≤ 0.85.
  - Old E's state: CEN 2.11, cascade area 2 1.33/1.37.
- **E2:** G3 out 30:40, 600 Mvar at IEEE 4 91:40, G3 back 152:20, line 3–4 out 241:00.
  - The line is picked by the declared rule (largest TS-a2 deficit in the end state, all
    mismatch gains < 2, frozen minimum ≥ 0.90 pu).
  - Stage bounds: TS-a2 +56.0 / −28.3 / −18.0 / −30.5 mpu; STS share — / 25 / 29 / 42 %.
  - Declaration `docs/analysis/2026-11-06_scenario_E2_declaration.md`, commit 0c80a62.
  - Runner check: dormant 600-Mvar load at bus 3, line 4 = IEEE 3–4, G3 committed, all four
    events registered on the grid.
- **Variant `Msel`** = M at the selected design's λ (1.405), run beside M (0.90).
  - STEP_ALLOWED (M, Msel) = {central_lambda_target}.
  - New test passes. The stale cooldown test still fails (unrelated).
- **Launched on e5power** via `run_ref_remote`:
  - E2: six variants, tag v4_E2, run 0025, remote launch 10:41:21.
  - Msel: tag v4_E2_Msel, run 0026, remote launch 10:44:57.
  - e5power got an empty 0024 (this host's CEN test).
  - Local placeholder `ref/0026_2026-11-06_104449_run_ref_reserved_for_e5power_0025_0026`;
    delete it after the pulls.
- **Results to Z:** (author: "copy over also to Z:/ ... only latest results").
  - `ref/0019_2026-11-05_215935_run_ref_v4_E_fixmt` copied to
    `Z:\Python_Projekte\qOFO\results\ch_11_case2_multisystem\ref\` (robocopy, 10:34).
  - Case 1 run 0001 follows once K1-PAR is done; E2 0025/0026 when back.
- **qOFO_2027** (author asked): the seven mentions in Ch 9 are the October comments on the old
  front and MC results, not written today. Today's runs use the worktree F:\qOFO_res of the qOFO
  repository.

## 11:57–12:07: interim results (author: "Progress?")

- **Status:**
  - E2 run 0025: O3, O2, M, S2 done; S1 and L2 still running on e5power.
  - Msel run 0026: pulled 11:29, ok.
  - CEN test 0024: ok (55 min).
  - F/FM reruns 0022/0023: only S1 still running.
  - Case 1 front: five variants still running.
- **CEN λ 0.90 on old E** (run 0024 vs M of 0019):
  - The period-2 cycle is gone. The reversal share of the TS-IBG movement in the 9–39 window
    is 0.03 (was 1.00).
  - e_v over all areas 7.20 mpu (was 7.64), AVR travel 0.094 pu (was 0.546).
  - Still below O3 (8.47).
- **E2, partial** (`sts_access_gain_scripts/e2_windows.py`, outputs
  `e2_windows_partial_1157.out` and `e2_windows_partial_1205.out`; variants pulled early into
  the scratchpad). Area-2 e_v per stage, mpu:

  | stage | O2 | O3 | O3 − O2 | S2 | M (0.90) | Msel (1.405) |
  |---|---|---|---|---|---|---|
  | 1 G3 out | 8.8 | 7.1 | −1.7 | — | — | 6.1 |
  | 2 +600 | 45.6 | 21.7 | −23.9 | 22.5 | 15.3 | 15.0 |
  | 3 G3 back | 18.4 | 13.9 | −4.4 | 16.0 | 13.8 | 13.7 |
  | 4 +3–4 | 27.0 | 16.8 | −10.2 | 16.8 | 17.3 | 17.3 |
  | all areas, whole run | 16.20 | 9.88 | | 13.44 | 9.21 | 9.14 |

  - **Hypotheses:**
    - H1: stage 1 |Δ| ≤ 2 mpu holds (−1.7).
    - H2: stages 2 and 4 strongly negative, holds; but stage 2 (−23.9) exceeds stage 4 (−10.2),
      against the declared ordering.
    - H3: stage 3 smaller, holds.
    - H4: CEN lowest in stages 2–4 holds in stages 2 and 3; in stage 4 CEN is 0.5 mpu above O3.
      CEN draws 2.5 % of the area-2 STS capability and leaves the level −5.6 mpu, while O3 draws
      24 % and restores it (−0.4).
    - H5: Msel shows no cycle while G3 is out (reversal share ≤ 0.17), against the static
      warning of 2.17–2.26.
  - Without access, O2's area-2 level falls to −30.9 mpu in stage 2 (bound −28.3).
- **λ rule:** on E2, M (0.90) and Msel (1.405) differ by ≤ 0.4 mpu per stage. On old E only 0.90
  is free of the cycle.

## 12:12–12:17: CEN ceiling back to the selected design's λ, 0.90 as fallback (author: "I would at least try 1.405 since it is the better argument, but if it does not work, take a lower lambda")

- **Rule:** M (CEN) is capped at the selected design's λ_max_B (1.405), the cascade's own ceiling.
  Lower it (variant M090, `M_LAMBDA_FALLBACK = 0.90`) only in a scenario where M runs a period-2
  cycle, as on old E (runs 0019/0024).
- **On E2,** M at 1.405 (run 0026, then named Msel) shows no cycle and has the lowest overall
  error (9.14 mpu).
- **Code** (`variants.py`):
  - M → selected λ.
  - Msel = M (kept for run 0026's provenance).
  - M090 = M at 0.90.
  - Guard pairs (M, Msel) = ∅ and (M, M090) = {central_lambda_target}.
  - Tests: 14 pass; the stale cooldown test still fails (unrelated, n = 44 vs 53).
- **Runs made while M was capped at 0.90** (09:44–12:15): 0024 (old E) and 0025/M (E2).
- **Plan for E2's CEN of record:** once 0025 is pulled, move 0025/M aside as `M_lambda090` and
  rerun M (1.405) into run 0025. The result should equal 0026/Msel (same config), which doubles
  as a determinism check.

## 14:42: Case 1 front complete; run 0001 copied to Z: (author: "you dont have to update the report, another session will do so")

- All 13 variants of `ch_10_case1_svr/0001_2026-11-05_215956_run_case1_v4_A` are ok
  (K1-REF + K1-PAR), and the manifest is complete.
- Copied to `Z:\Python_Projekte\qOFO\results\ch_10_case1_svr\0001_2026-11-05_215956_run_case1_v4_A\`
  (robocopy 14:42, 61 files, 122 MB, equal to the source).
- The run-level `metrics.csv`, `epost.csv`, `pareto.csv` and `equivalence.json` date from
  09:17 and cover the K1-REF variants only. Rerun `report_case1.py` on the run for the front.
  Per the author, another session does this.

## 18:43–18:45: E2 complete; CEN rerun at 1.405 into run 0025

- **Run 0025** pulled 17:13 (S1 done on e5power at 17:06), all six variants ok.
  - S1 slowed to ~46 s/step in stage 2 (11–12 coupler tap requests blocked by the rate limit,
    each followed by a second PF) and recovered after G3's return.
- **E2 per stage** (`e2_windows_full.out`), area-2 e_v differences in mpu:

  | stage | O3 − O2 | S2 − S1 |
  |---|---|---|
  | pre | −0.8 | +1.0 |
  | 1 | −1.7 | −3.0 |
  | 2 | −23.9 | −35.5 |
  | 3 | −4.4 | −5.4 |
  | 4 | −10.2 | −12.6 |

  - Whole-run e_v over all areas, mpu: L2 24.95, S1 21.38, O2 16.20, S2 13.44, O3 9.88,
    M(0.90) 9.21, Msel(1.405) 9.14.
- **F/FM vs E2 comparison:** `sts_access_gain_scripts/cmp_F_FM_E2.py/.out`.
- **CEN of record at 1.405:**
  - 0025/M (run at 0.90) moved aside as `M_lambda090`, with a NOTE.
  - M rerun into run 0025 locally: `run_ref --variants M --run-dir …0025…`, launched 18:44:29.
    Console: κ 1.268 → λ 1.405.
  - Expect ~1 h. It should equal 0026/Msel (same config).

## 19:35–19:40: CEN of record for E2 done; E2 copied to Z:

- **M (λ 1.405) rerun into run 0025:** finished 19:35, wall 3044 s, ok.
  - It reproduces 0026/Msel to round-off (max |ΔV| 1.4e-12 pu, max |ΔAVR| 1.3e-13 pu): the runs
    are deterministic.
  - metrics.json/report.md rewritten at 19:35. Whole-run e_v over all areas, mpu: L2 24.95,
    S1 21.38, O2 16.20, S2 13.44, O3 9.88, M 9.14, M_lambda090 9.21.
- **Copy to Z:** the background job's robocopy (called from bash) copied nothing (quoting).
  Redone with PowerShell at ~19:37. 0025: 39 files, 67 MB; 0026: 14 files, 10 MB; both equal to F:.
  `Z:\Python_Projekte\qOFO\results\ch_11_case2_multisystem\ref\0025_2026-11-06_104140_run_ref_v4_E2\`
  and `…\0026_2026-11-06_104505_run_ref_v4_E2_Msel\`.

## 21:24–21:45: shared checkout Z:\Python_Projekte\qOFO brought up to date (author: "please make Z:\Python_Projekte\qOFO up to date, commit and push"; "Pin runs to 08:00")

- **State found** (branch minimal/thesis-core):
  - HEAD f1f4a6c, not pushed.
  - 22 staged files, not this session's: coupler hold after a bank (`ch_9_13_coupler_hold.py`,
    its test, config field, base_controller, multi_tso_dso), live-plot overview, SCENARIO /
    VARIANT / TRIAL_PARAMSET in `run_multi_system_ofo.py`, trial and stage-1 paramsets, daily
    logs of 2026-11-02 to 2026-11-06.
  - The res branch was 120 commits ahead of the merge base, minimal/thesis-core 10.
  - The staged `make_config` moved the start to 06:00 and its default load events to bus 13
    (−350 Mvar at 90/330 min). `run_case1.base_config` inherited the start from `make_config`,
    so every Case 1/2 run of record would have moved to 06:00.
- **Steps:**
  1. Restored this session's own unstaged E2 entry in `scenarios.py` (the merge brings E2).
  2. Committed the staged work as it stood (0bf35ad).
  3. Merged `res/commitment-base-2026-10-31` (a383fe6). Two conflicts:
     - `ch_9_3_stage2_screen.py`: the same `--config-set` port on both sides; the branch's
       version, since dataclasses is imported at the top;
     - `_multi_tso_helpers.py`: the overview observables kept; the branch added nothing there.
  4. Pinned `run_case1.REFERENCE_START = 2016-01-05 08:00` in `base_config`. A scenario with its
     own period (GA, RA, ...) still replaces it.
- **Checks:**
  - `make_config` keeps 06:00 for the live plots. Case 1 on A, Case 2 E and E2 (also through
    `run_multi_system_ofo.scenario_config`) start at 08:00; GA at 2016-01-08 06:00.
  - The resolved configs of E2 (L2, S1, S2, O2, O3, M) and Case 1 on A (S1m, O0, O2, S1_T128)
    equal those of the runs of record (F:\qOFO_res code), field by field, apart from
    live-plot fields.
  - M at λ 1.405; fix_machine_taps in L2/S1.
  - Ten run/report/figure scripts import. 26 of 27 targeted tests pass; the stale cooldown test
    fails as before. Full suite: see the next entry.
- **Not touched:** main (169 commits behind). The default `make_config` contingencies (bus 13)
  now differ from the six of the old Case 1 K1-REF schedule; this matters only for
  `run_case1` without `--scenario` (Ch 9 uses scenario A).

## 21:52–23:55: full test suite, push

- **Full suite on the merged code:** 1008 passed, 19 skipped, 4 failed (1 h 39 min).
  - Two failures were the old versions of `test_m_is_ready_and_has_the_cascade_timescale` and
    `test_case2_configuration_and_noise_seed`; the run collected them before commit ec36288
    fixed both (13 of 13 pass).
  - The other two, `test_ch9_procedure_v2::test_engagement_bounds` and
    `::test_the_largest_rho_within_the_margins_is_chosen`, pass alone and in the first-42
    order. They fail whenever the whole suite is collected, on the pre-merge branch
    (F:/qOFO_res) as well: another test module changes the v2 procedure's module-level
    settings at import. Pre-existing, not caused by the merge; open.
- **Pushed** minimal/thesis-core to origin (ac69918..ec36288).
- Untracked files of another session that appeared meanwhile (2026-11-06c/d daily logs and
  hold-test scripts) are not committed.

## 2026-11-07, 01:05–01:25: two fixes applied, affected variants relaunched on e5wimi (author: "respect this for the ref runs and relaunch them for tonight? just use 6-10 workers here"; "only the specific variants please that are affected")

- **Read** the handovers in Z:\Python_Projekte\qOFO\docs\daily_log\11_2026:
  `2026-11-06e_HANDOVER_sim_session.md` (limiter fix and option, rerun list) and
  `2026-11-07_HANDOVER_svr_calibration.md` (SVR calibration fix).
- **Worktree F:\qOFO_res fast-forwarded** to minimal/thesis-core (dad2c60).
  - The untracked `configs/paramsets/stage1_v3_uc_2026_11_pq10.json` blocked it. It has the same
    JSON content as the tracked one (only CRLF differs) and was moved to the scratchpad
    `untracked_backup/`.
- **Applied unchanged and committed (c643d83):**
  - `2026-11-06d_limiter_fix_and_coupler_hold_area.patch` (rate limiter no longer reverts
    same-step relay moves; option `sts_coupler_hold_scope`, default 'host' = unchanged);
  - `2026-11-07_svr_calibration_qv_der.patch` (SVR calibration through the DER command path).
  - Tests: 92 pass (limiter, hold, cooldown, banks, Case 2 variants, SVR calibration/controller/STS
    access).
- **Affected variants:**
  - limiter: Case 2 L2, S1, O2; Case 1 all K1 variants (local coupler relays);
  - SVR calibration: Case 2 S1, S2; Case 1 S1, S1m, O0 (and the S1 front tunings).
  - O3 and M are not affected. Option A of the handover's O3 decision (no hold change) holds
    until the author decides; then O3 needs no rerun.
- **Relaunched** at 01:22:14 (pythonw, no console):
  - E2: S1, S2, L2, O2 rerun into run 0025 (`run_ref --run-dir`, 2 workers). The old four moved
    to `0025…/superseded_2026-11-07/` with a NOTE. O3, M and M_lambda090 stay.
  - Case 1 on A, all 13 variants (K1-REF + K1-PAR), new run
    `ch_10_case1_svr/0002_2026-11-07_012247_run_case1_v4_A_fix`, 8 workers.
  - Expected done around 07:30–08:30 (S1-type legs ~5–6 h).
- **Background watchers:** a smoke check when E2 S1 passes 112 min (couplers must now step after
  the load step at 91:40), and a copy of both runs to Z: when they end (PowerShell script
  `copy_reruns_to_Z.ps1`, file counts compared).
- **Correction, 01:24–01:26** (author: "13 ch9 variants? so many? you know the pareto front is not
  needeed anymore"):
  - The Case 1 run with K1-ALL (0002) was stopped about two minutes after its start (only its own
    process tree: launcher 109264, run_case1 160244, 8 workers). No variant was complete.
  - It was moved to `_local_archive/results/ch_10_case1_svr/` with a NOTE, so the number stays
    taken.
  - Relaunched with K1-REF only (L2, S1m, S1, O0, O1, O2), 6 workers, at 01:25:34:
    `ch_10_case1_svr/0003_2026-11-07_012602_run_case1_v4_A_fix`.
  - With E2's 2 workers, 8 are used. The copy watcher and script now point to run 0003.
- **02:48–02:55: smoke check of the limiter fix passed** (E2 S1, 91:40–112 min, from the
  console's `requested->kept` clamp lines).
  - Old run (superseded): every coupler's kept position flips between two values every 2 min
    (accepted, then reverted at the end of the step); 20 flips in 20 min, relays stationary.
  - Fixed run: kept positions step monotonically, one tap per 2-min lock (coupler 0: −2 → −3 → −4;
    coupler 5: −3 → … → −7).
  - S1 now runs ~3× faster (151 min simulated after 1.5 h), since the second PF per step for the
    blocked moves is gone.
- **05:20–05:30: reruns done and copied to Z:.**
  - Case 1 run 0003 (K1-REF) ended 04:07 (9688 s), E2 rerun into 0025 at 05:20 (14323 s);
    all variants ok.
  - The background copy failed (`powershell.exe` not on bash's PATH in the background job) and
    was redone at ~05:25 with the PowerShell tool.
    - 0025: 56 files, 109 MB; 0003: 29 files, 61 MB; Z: equal to F:.
    - `Z:\Python_Projekte\qOFO\results\ch_11_case2_multisystem\ref\0025_2026-11-06_104140_run_ref_v4_E2\`
      (the old four in `superseded_2026-11-07\`).
    - `Z:\Python_Projekte\qOFO\results\ch_10_case1_svr\0003_2026-11-07_012602_run_case1_v4_A_fix\`.
  - **E2 whole-run e_v over all areas, mpu** (before the fixes in brackets): L2 24.95 (24.95),
    S1 21.35 (21.38), S2 13.51 (13.44), O2 16.22 (16.20), O3 9.88, M 9.14 (unchanged variants).
  - **Case 1 run 0003:** S1m = O0 to 4.2e-10 pu (bus V) and 3.2e-12 pu (AVR).
    - e_v over all areas, mpu: L2 16.95, S1m/O0 13.53, S1 13.47, O1 10.21, O2 7.12.
    - Run 0001 had L2 16.96, S1m/O0 13.04, S1 12.91, O1 10.20, O2 7.18.
  - `report_case1.py` was not run on 0003 (author: another session does the reports).
