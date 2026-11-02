# Commitment base and GA runner merged into minimal/thesis-core (Z:)

Timestamp: 2026-11-02, ~10:55–12:00 Europe/Berlin. Author request: "please bring everything
to Z, by committing and fast forwarding or however you would do it".

## Starting state

- `Z:\Python_Projekte\qOFO`, branch `minimal/thesis-core` at f241a21, with the live-plot rework
  of the session "Live plotter redesign" uncommitted (26 modified, 6 new files).
- `res/commitment-base-2026-10-31` (worktree `F:\qOFO_res`, Ch 9 run 0004 running from it):
  75 commits ahead, 0 behind. It carries the merit-order unit commitment, the G1 Q limit, the
  Case-2 scenarios (A–E, GA, GE) and the Ch 9 procedure v2.
- `res/ga-runner-2026-11-02` (worktree `F:\qOFO_ga`): the commitment base plus bb7504d (the
  `SCENARIO` / `VARIANT` switch in `run_multi_system_ofo.py`, see `2026-11-02b_ga_selectable_in_runner.md`).

## What was done

1. **Held while the live-plotter session was writing to Z:** (last write 10:56).
2. **Trial merge** in a throwaway worktree, on a snapshot of the uncommitted live-plot files
   plus the GA branch:
   - nine files changed on both sides, all auto-merged;
   - `test_objective_terms.py` and `test_dso_voltage_band.py`: 8 passed;
   - GA/O3 smoke over 12 min passed.
   - The worktree and its `tmp/` branch were removed afterwards.
3. **The live-plot session committed its work** as 3f1577e. Z: was clean, and the session idle.
4. **`git merge --no-ff res/ga-runner-2026-11-02`** on `minimal/thesis-core`. A fast-forward was
   impossible, because 3f1577e and the 76 branch commits had diverged.
   - One conflict, in `docs/daily_log/INDEX.md`. The file is generated, so it was rebuilt with
     `docs/daily_log/_build_index.py`, giving 261 entries.
   - Auto-merged files changed on both sides: `configs/config.py`, the three controllers,
     `experiments/runners/multi_tso_dso.py`, `experiments/helpers/records.py`,
     `experiments/ch_10_case1_svr/run_case1.py`, `experiments/run_multi_system_ofo.py`,
     `tuning/stage1_screen.py`.

## Checks on the merged Z: tree

- **Removed live-plot flags.** No code still references `live_plot_system`, `_tracking` or
  `_sbx`. They remain only in one comment and in `run_params._FINGERPRINT_IGNORE`, where they
  are kept on purpose.
- **GA config.** `scenario_config("GA", "O3")` against REF run 0008 `O3/config_repr.txt`: only
  live-plot fields differ. Two windows are switched on, `live_plot_overview` and
  `live_plot_refresh_s` were added, and the three removed flags are gone. Visualisation only.
- **GA/O3 smoke**, 12 min, headless, all boolean live-plot flags off:
  - 36 records in 497 s;
  - G3 out of service (commitment);
  - V_min per zone 1.0071 / 1.0209 / 1.0146 pu, identical to the trial merge;
  - the new overview record fields are filled.
- **Tests:** `tests/test_objective_terms.py`, `test_dso_voltage_band.py`, `test_local_rules.py`,
  `test_tso_loss_objective.py` and `tests/tuning/`: **88 passed** (65 min on the network share).
  They ran on the tree that became merge commit 93020ae. The author asked for the commit before
  they finished, and it was made at ~11:50.

## Use

On Z:, set `SCENARIO = "GA"` (and `VARIANT`, default `"O3"`) in
`experiments/run_multi_system_ofo.py`, then run it from the project root with
`F:\python_environments\qOFO_clean\python.exe experiments\run_multi_system_ofo.py`.

## Open

- `F:\qOFO_ga` and `res/ga-runner-2026-11-02` are now contained in `minimal/thesis-core`, and
  the worktree can be removed.
- `F:\qOFO_res` is untouched; Ch 9 run 0004 still runs on `res/commitment-base-2026-10-31`.
  Later commits on that branch (e.g. the v2 result) must be merged into `minimal/thesis-core`
  again.
- The interactive live windows on GA have not been tried. The smoke run ran headless.
