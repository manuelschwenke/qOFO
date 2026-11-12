# 2026-11-12 – STS coupler hold after a sudden voltage dip (rate trigger)

Branch `res/coupler-dip-hold-2026-11-12`. It starts from 5c2d8b0, the code of the R2 reference run
0001_2026-11-10_155856_run_ref_v5rb_E2. The local worktree is `C:/Users/Manuel Schwenke/qOFO_runs/qOFO_rerun_dip`.

## Why

- In Ch 10 R2 (scenario E2), every area-2 coupler under OFO+ tapped down two or three times at the 600 Mvar load step
  (91:40). Within the next 35 min each tapped back up.
- Totals: 75 coupler operations against 31 net, and 17 direction reversals, 9 of them at this event.
- The taps act at once, because their gradient entries exceed g_w,OLTC. The IBGs take over gradually, and then the
  sign of the coupler gradient turns.
- The author asked for a hold of the couplers after a sudden dip, triggered by the voltage rate of change and released
  after 1 or 2 TS periods. The principle is that of tap-changer blocking (NC ER Art. 17(2)): continuous actuators
  act first.

## What changed

- **`configs/config.py`** (MultiTSOConfig):
  - New `sts_coupler_dip_hold_mpu` (float, 0 = off) and `sts_coupler_dip_hold_steps` (int).
  - Validation is in `validate_integrator_mode`: both must be ≥ 0, and a threshold > 0 needs steps > 0.
- **`experiments/runners/_multi_tso_helpers.py`**:
  - `coupler_bus_mean_vm(meas, hv_buses)`: mean measured voltage of the coupler HV buses in a DSO Measurement.
  - `coupler_dip_trigger(v_prev, v_now, threshold_mpu)`: one-sided; returns ΔV in mpu only for a dip larger
    than the threshold.
- **`experiments/runners/multi_tso_dso.py`**, DSO loop, right after `meas_dso` is formed:
  - Trigger signal: the mean measured TS-side voltage of the STS's interface couplers (`net.trafo3w.hv_bus`),
    compared with the previous STS step.
  - On a dip, `dso_ctrl.hold_oltcs(sts_coupler_dip_hold_steps)` runs. This is the existing zero-step bound of the
    tap lock, starting with this step. The DER stay free.
  - Each trigger is recorded in `rec.dso_coupler_dip_hold[dso_id] = dV_mpu` and printed as
    `[coupler-dip-hold ...]`.
  - New state `_dip_hv_buses` and `_dip_v_prev` before STEP 8.
- **`experiments/helpers/records.py`**: new field `dso_coupler_dip_hold` (dict, default empty, so old pickles load
  unchanged).
- **`experiments/ch_11_case2_multisystem/ref/run_ref.py`**:
  - New option `--config-set KEY=VALUE` (repeatable). It is applied to every variant and to the base config, which
    is validated.
  - The overrides are written to the manifest as `parameters.config_set`.
- **`tests/test_sts_coupler_dip_hold.py`**: config defaults and validation, trigger semantics, bus selection.
  Passes together with test_sts_coupler_hold(_area).py: 31 passed.

Defaults leave every existing run unchanged.

## Design choices

- **TS side, not STS side.** An own coupler step moves the TS-side voltage by about 2 mpu, but the STS mean by
  about 10 mpu. The trigger therefore does not react to the controller's own actions.
- **Threshold 20 mpu per STS step (20 s).** Screening of the R2/R3 OFO+ logs (thesis repo,
  `00_daily_log/2026-11-12_ch10_r2_discrete_lanes.md`):

  | Case | Step (mpu) |
  |---|---|
  | Normal steps incl. TS dispatches | ≤ 8.9 |
  | Line trip (4) | 11–15 |
  | Load step (2), STS 1–3 | 57–68 |
  | Load step (2), STS 4 | 23 |

- **One-sided:** rises never fire.
- **Per STS:** the "local" method of NC ER Art. 17(2)(a).
- **Release by time only:** 9 or 18 STS steps, i.e. 1 or 2 TS periods. There is no early release on the voltage band
  or on IBG headroom; the STS MIQP keeps its voltage constraints for the DER. A new dip restarts the hold.

## Runs

- **Smoke:** `results/.../0001_2026-11-12_114337_run_ref_smoke_diphold` (5 min, threshold 0.3 mpu). The hold fired at
  200 s for DSO_1 to DSO_3 and was recorded. The manifest has `config_set`, plus `code_state.diff` because the tree was
  uncommitted.
- **R2 OFO+, threshold 20 mpu**, tags `v5rb_E2_diphold20_h9` and `v5rb_E2_diphold20_h18`. Launch logs are in
  `qOFO_runs/launch_logs/2026-11-12_r2_diphold_h{9,18}.log`. Results: see below.

## Results

Runs: `0002_2026-11-12_114514_run_ref_v5rb_E2_diphold20_h9` and `0003_2026-11-12_114536_run_ref_v5rb_E2_diphold20_h18`.
Each took 16 min on this PC. Both are uncommitted, with code_state.diff in each run folder.

**Triggers.** In both runs the hold fired only at 91:40, for DSO_1/2/3, with measured dips of −41.9, −51.7 and
−38.9 mpu. It did not fire for DSO_4, nor at (1), (3) or (4). Comparison script: thesis session scratchpad
`r2_diphold_compare.py` / `.out`. Diagnostic plot: `r2_diphold_lanes.png`.

| | base O3 | h9 | h18 | CEN |
|---|---|---|---|---|
| coupler ops / reversals / net | 75 / 17 / 31 | 55 / 17 / 31 | 45 / 12 / 31 | 0 |
| area-2 tap sum, minimum in the detour | −16 (96 min) | −5 | −1 | – |
| STS V min, area 2, 91.67–135 min [pu] | 0.9317 | 0.9250 | 0.9250 | 0.9990 |
| bus 4 min [pu] | 0.9218 | 0.9258 | 0.9258 | 0.9785 |
| λ_min(J_R) min | 0.358 | 0.366 | 0.366 | 0.393 |
| mean Σ\|Q_set − Q_act\|, area 2, 91.67–135 min [Mvar] | 59.5 | 52.3 | 42.3 | – |
| e_v RMS over areas [mpu] | 9.84 | 9.78 | 9.70 | |
| area-2 e_v / worst [mpu] | 14.64 / 108.2 | 14.52 / 104.2 | 14.39 / 104.2 | |
| W at (2) [Mvar min] | 218.8 | 175.0 | 175.0 | |

- **During the hold:** the area-2 export is 20–40 Mvar higher, and bus 4 is up to 6 mpu higher. The STS load P stays
  lower; the load relief is kept for the length of the hold.
- **After release:** the h9 couplers tap down one step each at 94:40 and back up later, so 17 reversals remain.
  Under h18, five area-2 couplers (t4, t5, t7, t8 and t1/t3 partly) tap down one step at 97:40 and back up later.
  So the down-up pattern is much smaller but not fully gone.
- **Cost:** the area-2 STS mean voltage reaches 1.03 pu about 4 min later than in base, and the STS minimum is
  7 mpu lower (0.925 pu, within the 0.90 band). DSO_2 f_V rises from 18.71 to 19.01 mpu.
- **Everything outside (2) is identical:** the per-STS coupler count for DSO_4 is 7 in all runs, and the events at
  (1) are unchanged.

### h27 (3 TS periods, author request; run `0004_2026-11-12_120939_run_ref_v5rb_E2_diphold20_h27`, 13.8 min)

Same triggers as h9 and h18.

| | base | h18 | h27 |
|---|---|---|---|
| coupler ops / reversals (all) | 75 / 17 | 45 / 12 | **39 / 5** |
| area 2 ops / reversals | 68 / 15 | 38 / 10 | 32 / 3 |
| area-2 tap sum: lowest after (2) → end | −16 → +17 | −1 → +16 | +3 → +15 |
| area-2 STS mean at 98 / 100 / 105 min [pu] | 1.038 / 1.039 / 1.040 | 1.023 / 1.031 / 1.037 | 1.015 / 1.022 / 1.033 |
| area-2 STS worst min at 98 / 100 min [pu] | 1.009 / 1.013 | 0.987 / 1.002 | 0.975 / 0.981 |
| run STS min | 0.932 | 0.925 | 0.925 (during the hold, same in all hold runs) |
| bus 4 min / λ_min(J_R) min | 0.922 / 0.358 | 0.926 / 0.366 | 0.926 / 0.366 |
| area-2 Σ\|Q_set − Q_act\|, window / run [Mvar] | 59.5 / 41.2 | 42.3 / 40.6 | 36.3 / 35.2 |
| e_v RMS areas; area 2 e_v; area 2 t_out | 9.84; 14.64; 0.7 % | 9.70; 14.39; 0.6 % | 9.75; 14.47; 1.4 % |
| STS f_V DSO_1/2/3 [mpu] | 14.96 / 18.71 / 13.69 | 14.98 / 19.01 / 13.69 | 14.85 / 18.41 / 12.92 |
| coupler ops DSO_1/2/3/4 | 21 / 24 / 23 / 7 | 11 / 14 / 13 / 7 | 9 / 13 / 10 / 7 |

- **Area 2 after release:** only STS 2 t5 taps down once (100:40) and stays there. Every other area-2 coupler moves
  only up, which is the net requirement. So the down-up of Fig. 10.4 is gone in area 2.
- **STS 4:** t11 still has a one-step down-up. STS 4 is not held, because its measured dip stays below 20 mpu.
- **Cost:** the area-2 STS minimum stays below 1.00 pu until about 101 min (base: about 96 min). It never goes below
  0.925 pu (band 0.90). The area-2 t_out rises from 0.7 % to 1.4 % of the samples.
- **Not yet in the code:** an early release when an STS voltage nears its band, or when the IBG headroom in the needed
  direction is exhausted. This is recommended before the MC.

Open: adopting the hold into OFO+ would need every OFO+ run re-run (R2, R3, MC, and possibly Ch 9). This is the
author's decision.
