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
