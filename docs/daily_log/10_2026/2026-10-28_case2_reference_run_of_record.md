# 2026-10-28: Case 2 reference run of record (0009) with the adopted M

Branch `case2/ref` (worktree `.claude/worktrees/case2-ref`). Continues
`2026-10-27e_case2_reference_run_infrastructure.md` (M adoption section).

## Run

- **Run:** `results/ch_11_case2_multisystem/ref/0009_2026-10-28_093739_run_ref`.
- **Setup:**
  - code 06d403b, clean; paramset `ch9_selected_band_2026_10`;
  - scenario A, 360 min, noise "minimum", seed 20261027;
  - e5power, one process per variant, started 09:37, finished 13:19.
- **Outcome:** all six variants `ok`, 1080 records each.
- **Copy:** pulled to the share at the same relative path. All 32 files were checked by SHA-256
  against e5power.
- **Figures:** `figures_ref.py` wrote vtrack, qtrack and tieflow (PDF, PNG and CSV) to
  `figures/`.
- **Archived:** `0006_2026-10-27_181840_run_ref` (old M, κ 2.42) moved to
  `_local_archive/results/...` with status `superseded`, `superseded_by` 0009.
- **Kept:** 0007 (`m_tuning`, the four M candidates) stays on the share as diagnostic evidence.
  0008 (`m_check`) stays on e5power only.

## Reproducibility (fact)

L2, S1, S2, O2 and O3 are unchanged from 0006 (code 20873f4 → 06d403b):

- every metric in `report.md` is identical apart from the M rows;
- L2's `log.pkl` is byte-identical;
- the other logs differ only in `zone_tso_solve_s`, the MIQP wall time, which is not a model
  quantity.

So the M changes leave the other variants untouched.

## Results (mpu; aggregate = RMS over the three areas)

| | L2 | S1 | S2 | O2 | O3 | M (0009) | M (0006, old) |
|---|---|---|---|---|---|---|---|
| e_v | 22.01 | 12.73 | 11.37 | 8.07 | 6.83 | **6.25** | 7.79 |
| e_vp | 17.92 | 4.27 | 2.98 | 4.58 | 3.46 | **2.98** | 3.66 |

M against O3 is −8.5 % in e_v and −13.9 % in e_vp. Per area (e_v / e_vp):

| Area | M | O3 |
|---|---|---|
| 1 | 4.15 / 4.81 | 4.64 / 4.62 |
| 2 | 9.26 / 1.55 | 9.84 / 2.84 |
| 3 | 3.77 / 1.06 | 4.64 / 2.55 |

Effort, M against O3:

| | M | O3 |
|---|---|---|
| TS tap moves | 11 | 3 |
| Bank switchings | 8 | 5 |
| AVR travel (pu) | 0.32 | 0.09 |
| Coupler tap moves | 1 | 2 |

STS voltage band: in O3, DSO_4 leaves 0.92–1.08 by 4.9 mpu (as in 0006). M keeps every STS inside
the band; its smallest margin is DSO_2 at 3.4 mpu.

## Observations

**Facts:**
- M reacts to the generator trip and the load step at the event step: its taps and banks move at
  30:40 and 121:40. O3 moves them at the next dispatch (33:00, 123:00).
- The 18-min lock never binds in scenario A: the shortest gap between two moves of one device is
  30 min.
- After the load step (121:40), M's area-1 error rises by 6.2 mpu within the window to the next
  180-s grid instant, while area 2 improves by 9.5 mpu. It decays in about 10 min (vtrack).
  O3's area 1 rises by 0.3 mpu only.
- O3 after the load step: area 2 +2.8 mpu, W = 55 Mvar·min. This is the inter-dispatch withdrawal
  (H-NMP), unchanged from 0006.

**Hypothesis:** M trades area 1 against area 2 because its objective is one weighted sum over all
areas. The zone controllers cannot do this, since each controls its own area. To be checked in
the step tests (load step in one area, error of the other areas).

## Code changes after the run (wording only; no behaviour change)

- `variants.py`: M is a "centralised reference at equal weights and equal device rate limits",
  not a "bound" (author 2026-10-28). The docstring of `selected_lambda_max` now describes the
  ceiling rule.
- `figures_ref.py`: docstring wording.
- Tests: `tests/experiments/test_case2_variants.py` and `tests/test_central_updater_wrapper.py`,
  12 passed.

## Open

- Thesis: paste `latex_header.tex` under `ch:case3:nominal`. Copy `figures/data/*.csv` to
  `graphics/Ch11/data/2026-10-28_case3_ref/`. Render the label numbers on e5wimi
  (`QOFO_THESIS_DIR`) and run `thesis_runs check`.
- Discuss the O3 inter-dispatch withdrawal and the O3 DSO_4 band violation before any change.
- Plan the step tests (incl. load ramps) and the Monte-Carlo.
