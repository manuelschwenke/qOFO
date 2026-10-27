# Ch 9 results brought under the 2026-10-23 layout: rename, run of record, archive, thesis provenance

Timestamp: 2026-10-27, ~13:45–15:00 Europe/Berlin.

**Reason:** the author asked whether the weight-selection campaigns followed the
results guideline of 2026-10-23 (`2026-10-23_chapter_results_layout.md`). They
did not:
- the campaigns sat as top-level `results/THESIS_ch8_*` folders, with no
  README, manifest or LaTeX header;
- the CDF follow-up had overwritten data in the September-dated thesis folder.

The author approved four steps, to run after the margin check: rename the Ch 9
producer folder, adopt the band campaign as the run of record, archive the
superseded campaigns, and fix the thesis data folder and provenance blocks,
with no commit in the thesis.

## STS voltage-margin check (finished 13:43, before the moves)

`margin_check/margin_report.txt` in the run of record compares the selected
weights fde2e81af6b1 with the Stage-I point. Values are the extremes over
W1–W4 and W6, with the Stage-I value in brackets:

| STS | v_min | v_max | margin to 0.90/1.10 | min outside 0.92–1.08 | e_v(DS) | coupler taps/day |
|---|---|---|---|---|---|---|
| DSO_1 | 1.0039 (1.0078) | 1.0677 (1.0596) | 0.032 (0.040) | 0 | 0.0079 (0.0052) | 0 (4) |
| DSO_2 | 1.0200 (1.0157) | 1.0778 (1.0548) | 0.022 (0.045) | 0 | 0.0129 (0.0061) | 0 (4) |
| DSO_3 | 1.0101 (1.0165) | 1.0617 (1.0580) | 0.038 (0.042) | 0 | 0.0067 (0.0058) | 0 (2) |
| DSO_4 | 1.0086 (1.0084) | **1.0823** (1.0693) | 0.018 (0.031) | **176 (W1)** | 0.0192 (0.0104) | 2 (4) |

- **Reproduction:** 10 / 10 re-simulations reproduce the stored e_v(TS)
  exactly.
- **TS:** e_v(TS) 0.00439 vs 0.00545; TS margin 0.051 pu.
- **Band:** the soft band is exceeded once, by up to 0.0023 pu, for 176 min.
  This happens in DSO_4 in W1, the generator-trip window.
- **STS coupler taps:** they barely act at the selected weights (0–2 per day,
  against 2–4).
- **Interface-Q tracking RMS:** 1.4–4.0 Mvar vs 6–24 Mvar.

## What changed (qOFO)

| File | Change |
|---|---|
| `experiments/ch_9_parameter_selection/` → `experiments/ch_09_parameter_selection/` | `git mv`, the deferred rename of `docs/architecture/chapter_folder_map.md`. Old name replaced in the folder's modules, README and RUNBOOK; `pf/timescale_study.py`; `tuning/stage1_screen.py`; `tuning/settling_predictor.py`; `experiments/run_multi_system_ofo.py` (3 import lines; only these hunks are committed, the author's uncommitted edit stays uncommitted); `tests/experiments/test_ch9_*.py`; `.gitignore`. **Not changed (historical records):** `docs/data/**`, `docs/handover*`, `docs/handoff/`, daily logs, and the provenance fields of `configs/paramsets/*.json`, whose sha256 is recorded by runs. |
| `configs/config.py`, `tuning/stage2_search.py` | The two stale path comments the folder map deferred to the rename: `part3_mc` → `ch_11_case2_multisystem/mc`; `ch_10_case_study/ch_10_1_variant_ladder.py` → `ch_11_case2_multisystem/ladder/variant_ladder.py`. |
| `experiments/thesis_runs.py` | New `adopt_run(source, *, experiment, producer, script, target, invocations, primary=0, …)`, also as CLI `adopt <dir> --spec <json>`. It moves a finished pre-layout campaign to `results/<chapter>/<NNNN>_<first start>_<producer>[_<tag>]/` and builds the manifest from the campaign's own records: commit and diff at launch, start and end times, not the current tree. The manifest records `adopted: {from, on}`, and the README status says "adopted from". `primary_invocation` lets the header name the run that produced the results (the driver) rather than the last follow-up. `_next_counter` factored out of `_new_run_folder`. |
| `tests/experiments/test_thesis_runs.py` | +4 tests: adoption (move, records from the campaign, README and header); counter continuation and refusals; failed status; primary invocation in the header. |
| `tests/experiments/test_ch9_timescale_seperation.py` | A stale assertion, unrelated to the rename: `d867287` renamed the heading to "Measured but not in the settling table", and the test still expected "Table 9.1". |
| `experiments/README.md`, `docs/architecture/chapter_folder_map.md` | Rename recorded as done. |
| `docs/tuning/2026-10-22_*`, `2026-10-24_*`, `2026-10-25_*`, `weight_selection_method_overview.md` | Location notes pointing to the run of record and the archive. Frozen text unchanged. |

## Results tree (not in git)

- **Run of record:** `results/ch_09_parameter_selection/0001_2026-10-25_215433_weight_selection_driver_band/`,
  adopted from `results/THESIS_ch8_weight_selection_band_20261025/`.
  - Contents: `README.md`, `run_manifest.json`, `latex_header.tex`,
    `adoption_spec.json`.
  - Invocations: the driver at 9d03c34 (primary), then the three follow-ups
    at d9633fe, all with the same dirty diff `code_state/tracked_changes.patch`.
  - Targets: `ch:timescales:weights`, `ch:timescales:period:confirmation`.
  - Feeds: `tab:param:weights:resolved`, `fig:param:timescales:settling_cdf`,
    `tab:param:timescales:ninner`.
  - Exports recorded: the thesis CDF files.
  - The scripts inside are the record of what ran. They resolve the
    repository from their old position and import the old folder name, so
    they are not re-runnable in place; the README says so.
- **Archived** with `thesis_runs archive`, each with `ARCHIVED.md` naming the
  run of record:
  - `results/THESIS_ch8_weight_selection_20261024/` →
    `_local_archive/results/THESIS_ch8_weight_selection_20261024/`
  - `results/THESIS_ch8_expB_banks_20261022/` →
    `_local_archive/results/THESIS_ch8_expB_banks_20261022/`
- **Index:** `results/README.md` (new "Moved on 2026-10-27" table, old → new
  paths) and `results/ch_09_parameter_selection/README.md` (runs of record).
- **Left in place:** `results/stage1/operating_points` (input of the run of
  record) and the 2026-10-22 bank-check folders.

## Thesis clone `Z:\latex_diss_clone` (uncommitted, as requested)

- `graphics/Ch9/data/2026-09-30_settling/settling_cdf.dat` restored with
  `git checkout`. Its README gets a "superseded 2026-10-27" note; the folder
  stays the record of the old figure.
- New `graphics/Ch9/data/2026-10-27_settling/`:
  - `make_settling_cdf.py`: the September generator with its input pointed
    at the run of record's `expC/` (fixed run stamps instead of a glob).
  - `settling_cdf.dat`.
  - `README.md`: sources and a numbers table, p95 25.0 vs 31.1 for the
    starting weights.
  
  The generator rewrote `graphics/Ch9/settling_cdf.tex`: F(27) 0.967 vs
  0.700; F ≥ 0.95 first at n = 25 vs 31.
- `Chapters/Chapter09.tex`:
  - The `RESULT PROVENANCE` block of the run of record is pasted under
    `\section{Selection of the Controller Weights}` and
    `\section{Confirmation of the Period at the Final Weights}`.
  - The source comments and `\todo`s now name the run of record and
    `ch_09_parameter_selection`.
  - Following the author's own comment ("regenerate …, then restore"), the
    figure `\input{graphics/Ch9/settling_cdf}` is restored and the `[TBD]`
    placeholder box removed.
  - The `[TBD]` result paragraphs are not written.

## Verified

- `python -m experiments.thesis_runs check --thesis Z:\latex_diss_clone`:
  2 blocks, 0 failing, 0 warnings.
- `tests/experiments/test_thesis_runs.py`: 15 passed.
- The renamed modules import.
- Full suite: 902 of 903 tests passed (874 in the full run, plus the 28 it had
  not reached, run separately). The last test,
  `tests/tuning/test_runner.py::test_run_one_handles_bad_params_gracefully`,
  is a 1-h simulation at `g_w_der = 1e-6` that does not touch the changed
  code. It was still computing at commit time (> 20 min).
- A first full run was killed after 27 min at 1.7 s CPU (blocked; cause not
  identified); the rerun computed normally.

## Open

- **Another session's worktree** `.claude/worktrees/case2-ref` (branch
  `case2/ref`, from 04b0d42) adds
  `from experiments.ch_9_parameter_selection._ch9_selected_design import …`
  to `experiments/ch_10_case1_svr/run_case1.py`, uncommitted. After it merges
  onto the renamed tree, that import must read `ch_09_parameter_selection`.
- `ch:timescales:period:confirmation` has no number in the clone's `.aux`
  files, since the thesis was not rebuilt after the 2026-10-26 rewrite.
  Harmless; it resolves on the next build.
