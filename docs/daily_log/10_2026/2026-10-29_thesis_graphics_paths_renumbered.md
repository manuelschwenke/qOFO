# Thesis graphics paths follow the renumbered chapters

**Date:** 2026-10-29

## Reason
The thesis `graphics/` subfolders were renumbered to the current chapters
(see the thesis log `00_daily_log/2026-10-29_graphics_folders_renumbered.md`):
Ch6->Ch5, Ch7->Ch6, Ch8+setup->Ch7, Ch9->Ch8, case_svr->Ch9, Ch11->Ch10, and the
orphaned ladder folder Ch10 -> `graphics/archive/Ch10_ladder_2026-09-03`.
One qOFO export script still wrote to `graphics/Ch11`; a re-run would have
re-created the retired folder.

## Changes
- `experiments/ch_11_case2_multisystem/ref/figures_ref.py` (**case2/ref
  worktree only**, the file is not on `minimal/thesis-core`): `_to_thesis`
  export target `thesis / "graphics" / "Ch11"` -> `"Ch10"`; the same change in
  the module docstring, the `--thesis` help text, the README it writes and the
  two `record_export` paths in the run manifest.
- `experiments/ch_08_setup/reference_period/characterise_reference_period.py`
  (both checkouts): docstring example `--fig-dir <thesis>/graphics/setup` ->
  `graphics/Ch7`. The folder is an argument, so behaviour is unchanged.
- `experiments/ch_11_case2_multisystem/ladder/figures.py` and
  `export_tikz_data.py` (both checkouts): comments only, pointing at the
  archived ladder folder and at `graphics/Ch8/data/`.

## Not changed
- Run folders under `results/` (provenance), e.g. the hard-coded
  `latex_diss_clone\graphics\Ch9` in
  `results/ch_09_parameter_selection/0001_.../followup_settling_cdf.py`.
- Nothing committed: the case2/ref edits sit uncommitted on that branch's
  worktree beside its running STEP/MC campaign.

## Verification
- `ast.parse` of the edited files passes; a search finds no other hard-coded
  `"graphics" / "ChN"` target in `experiments/` of either checkout.
