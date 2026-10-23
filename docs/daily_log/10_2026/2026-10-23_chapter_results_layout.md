# Thesis results organised per chapter; every run documents itself

Timestamp: 2026-10-23, ~12:30–14:30 Europe/Berlin.

Reason: the evidence behind the dissertation lay in at least five places
(`qOFO_GH/results`, `qOFO_2027/results`, `qOFO/results`, `qOFO/docs/data`,
the thesis's `graphics/<ChN>/data`). Several folder names carried chapter
numbers that no longer matched: the thesis was renumbered on 2026-09-22,
2026-10-06 and 2026-10-23. Author decision of 2026-10-23:

- results per chapter, mirrored from `experiments/`, folders
  `ch_NN_<description>` (zero-padded);
- a README in every run folder (date, save location, parameters, free
  description, time-series window, purpose in the thesis);
- an identically structured `RESULT PROVENANCE` comment block in the thesis
  under the heading that reports a result;
- older evidence stays where it is, behind pointer READMEs, and is re-run into
  the new tree over time;
- superseded runs **move** to `_local_archive/results/<same path>`, never
  deleted.

## What changed

| File | Change |
|---|---|
| `experiments/thesis_runs.py` (new) | `start_thesis_run(__file__, target=<label>, feeds, config, paramset, parameters, time_series, note, tag, resume)` derives `results/<chapter>/[<sub>/]<NNNN>_<stamp>_<script>[_<tag>]/` from the script's own path and refuses a script outside a `ch_NN_*` / `app_x_*` folder. It writes `README.md`, `run_manifest.json`, `latex_header.tex`, `config.json/.pkl`, and `code_state*.diff` when the tree is dirty. It is a context manager: normal exit = `complete`, exception or `run.fail()` = `failed`. `resume=` appends an invocation (campaigns that grow later). It records the label numbers "as of" the run from the thesis `.aux`. `annotate_run` records outputs and exports added by report/figure scripts. `latest_run` filters by producer, tag and status. `archive_run` moves a run to `_local_archive/results/` and marks it `superseded`; archived counters stay taken. CLI: `header`, `archive`, `check` (verifies every provenance block in the thesis: run exists, `complete`, not superseded, labels defined). |
| `experiments/reference_period/` → `experiments/ch_08_setup/reference_period/` | `characterise_reference_period.py`: `--run-dir` defaults to the newest complete `run_case1` run; without `--out` it writes a documented run (target `ch:setup:scenarios:reference`); `--fig-dir` exports are recorded. Body moved into `characterise()`. |
| `experiments/case1_svr/` → `experiments/ch_10_case1_svr/` | `run_case1.py` writes through `start_thesis_run` (target `ch:case_svr`); its own `_provenance()` removed; `--run-dir` resumes an existing run; a failed variant marks the run `failed`. `report_case1.py` and `figures_case1.py` default to the newest complete run and record what they wrote. |
| `experiments/part3_mc/` → `experiments/ch_11_case2_multisystem/mc/` | `run_mc.py`: one documented run folder per campaign. Without `--new` a call continues the newest campaign with the same tag, as an invocation in its README. `--smoke` = tag `smoke`. `--out` stays as an undocumented escape hatch; `PART3_MC_OUT` removed. `_provenance()` replaced by `_campaign_parameters()` + `elements.json`; main loop moved into `_simulate()`. `area1_pilot_weight.py` likewise. `README_server.md` updated; `HANDOVER_server_claude.md` marked historical. |
| `experiments/ch_10_case_study/` → `experiments/ch_11_case2_multisystem/ladder/` | the superseded seven-rung ladder, files without `ch_10_N_` prefixes. Producers write documented runs. **`figures.py` and `export_tikz_data.py` no longer write into the thesis by default** (`<run>/figures`, `<run>/tikz_data`; `--out` to export), because `graphics/Ch10` no longer belongs to this design. `ladder_mc.py` passes its run folder to spawned workers via `LADDER_MC_OUT`. |
| `sys.path` roots | `run_mc.py`, `ladder_mc.py` (`parents[1]` → `parents[2]`) and `preview.py` (`parents[2]` → `parents[3]`): the move put them one level deeper. |
| `experiments/ch_12_case3_sts_support/` (new) | README only; no producer yet. |
| READMEs | `experiments/README.md` (chapter table; duplicate closing paragraph removed), one README per new chapter folder, `ladder/README.md`; `results/README.md` (index, layout, archive rule, unassigned folders) and `results/ch_*/README.md` (pointers to the older evidence, marked "cited" or "by name"). |
| `docs/architecture/chapter_folder_map.md` | status updated in place; naming question marked decided. |
| `tests/experiments/test_thesis_runs.py` (new, 11) | mirror path and README/header fields; refusal outside a chapter folder; failed status on exception and on `fail()`; fingerprint ignores `result_dir`; hand-edited description survives; resume; archive move + counter; `latest_run` filters; `check` accepts a pasted header, flags an archived run and an undefined label. |

## Deliberately not changed

- **`experiments/ch_9_parameter_selection/`** is not renamed: the Experiment B
  bank campaign on another machine (`results/THESIS_ch8_expB_banks_20261022/driver.py`)
  calls three of its scripts by path. The call sites for the later rename are
  listed in `docs/architecture/chapter_folder_map.md`.
- `experiments/results_io.py`, `experiments/paths.py`, `configs/`, `tuning/`,
  `experiments/run_multi_system_ofo.py`: imported by that campaign's
  subprocesses. `thesis_runs.py` is a new module for that reason. The stale
  path comments in `configs/config.py:361` and `tuning/stage2_search.py:258`
  wait for the rename.
- No `git mv`, no staging: another session commits in this repository. Git
  sees deletions plus untracked folders until the author stages them.

## Verified

- `pytest tests/experiments`: 43 passed (11 new, 32 existing Ch 9).
- All 14 moved modules import.
- Smoke runs `run_case1 --only L2 --minutes 10 --tag smoke` (twice): the run
  folder, README, header, diff and manifest were written as specified. The
  second run is `failed` after `run.fail()` was added. Both were archived with
  the CLI to `_local_archive/results/ch_10_case1_svr/`.
- `python -m experiments.thesis_runs check` against the thesis: 0 blocks, 0
  failures (no block pasted yet).

## Open

- **No producer of Ch 8/10/11 can make a run of record today.** `ch9_selected`
  has no `g_w_tso_shunt`, a required weight since 2026-10-22 (see
  `2026-10-22_bank_price_searched_weight.md`). The smoke run stops there for
  every variant.
- The thesis has no `RESULT PROVENANCE` block yet. The pointers in
  `results/ch_*/README.md` list which older runs each chapter cites.
- The Ch 9 producers are not wired into `thesis_runs.py` (deferred rename).
