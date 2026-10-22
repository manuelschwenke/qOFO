# Stage-I screen counts one bank per STS; Experiment B with banks launched overnight

Timestamp: 2026-10-22, ~22:30 Europe/Berlin.

Reason: author decision after the Δv_eng,sh table of
`2026-10-22_bank_price_searched_weight.md`: the screen with every bank column
counted the MSC and the MSR of one STS as moving together, which the MSC/MSR
exclusion forbids. Then: "run the full tuning suite over night" — the
remaining protocol decisions were set by Claude under that delegation and are
marked for confirmation in `docs/tuning/2026-10-22_protocol_experiment_B_banks.md`.

## What changed

| File | Change |
|---|---|
| `tuning/stage1_screen.py` | `LayerCurvature.exclusive_groups` (column groups of which at most one moves per direction). `_tso_curvature(..., bank_sts)` forms one group per STS from the bank buses; `build_operating_point` passes the bus→STS map from `meta.hv_networks` when `shunt_dispatch='miqp'` and `shunt_sts_mutual_exclusion`. `lambda_max_B` keeps the strongest member of each group, by its own contribution `C_ii = ||a_i||²/g_w,i` (new helper `_exclusive_keep`); this reaches the screen, the integer floor and the Stage-I bisection alike. `_network_key` carries the exclusion flag. Module docstring updated. |
| `tests/tuning/test_stage1_banks.py` | New test: an exclusive group screens as its stronger column alone, and below the ungrouped pair; cache key separates exclusion on/off. |
| `configs/paramsets/stage1_start_10_2026_banks.json` | New, written by `tuning.stage1_screen --design --frozen-network --d-engage-ts 0.0075 --d-engage-sts 0.015 --d-engage-shunt 0.010 --write-paramset`. |
| `docs/tuning/2026-10-22_protocol_experiment_B_banks.md` | New protocol, frozen 22:45. |

## Results so far

- Operating points rebuilt (key `net82afed5b`); groups captured as intended:
  TS_2 three MSC/MSR pairs (DSO_1–3), TS_3 one (DSO_4), TS_1 none.
- Table with the reduction in the code
  (`results/2026-10-22_ch8_shunt_engage_table/engage_table.log`) reproduces
  the hand-reduced variant exactly: Δv_eng,sh 0.005 → 13.6, floor 0.85,
  κ_TS 0.44; 0.0075 → 20.4, 0.59, 0.126; **0.010 → 27.2, 0.50 (taps), 0.093**;
  0.015 → 40.8, 0.50, 0.074. `ch9_selected_10_2026`: λ_max(B) 2.18 at a bank
  price of 10, 1.47 at 27.2.
- Stage I at 0.010: `g_w_tso_shunt` 27.2121, `g_w_der` = `g_w_pcc` 0.09311
  (+0.25 decade over the October start), all other weights unchanged; worst
  λ_max(B) 1.000 (TS_2, W5).
- Stage II dry run: 119 / 512 Sobol draws pass the screen (23.2 %) → 120
  candidates × 6 windows × 12 h = 720 units, 30 workers.

## Campaign

`results/THESIS_ch8_expB_banks_20261022/driver.py` runs Stage II → selection
(bank budget 12/day, sensitivity 4–48) → Experiment C with the repeated
selection of Amendment 3 → writes `ch9_selected_10_2026_banks` (not adopted as
`SELECTED_PARAMSET`). Orientation run of `ch9_selected_10_2026` at the
Stage-I bank price alongside Experiment C. Progress in `driver.log`; outcome in
`DONE.txt` or `FAILED.txt`; record in protocol §11.

## Verification

- `tests/tuning/test_stage1_banks.py`, `test_stage2_shunt_wear.py`,
  `test_parameters.py`: 33 passed. The full suite was not re-run after this
  change (last full run, before it: 867 passed); it would compete with the
  campaign for the CPU.
- Driver's repeated-selection logic checked offline with stubbed
  Experiment C verdicts (band re-anchoring after a failure; no winner when all
  fail).
