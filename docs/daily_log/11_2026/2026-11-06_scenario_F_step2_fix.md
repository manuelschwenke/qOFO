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
