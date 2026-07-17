# SBX-H compact TU Darmstadt live-plot layout

**Timestamp:** 2026-07-13 (Europe/Berlin)  
**Reason:** Manuel requested removal of the status-text/title collision,
a tighter experiment-014 layout, and consistent use of the TU Darmstadt
palette with 5c yellow-green as the primary colour.

## Visual decision

- `TU_PRIMARY` is fixed to 5c yellow-green (`#B1BD00`).
- The primary colour is used for the figure header and the dominant
  first series (corridor residual, side-A voltage, and area-1 payment).
- Semantic colours remain distinct: 3c teal denotes hold, 9c red denotes
  violation, neutral gray denotes transition, and 10c magenta denotes
  paid signed reactive support.

## Code changes

- `configs/color_config.py`
  - central TU Darmstadt palette;
  - named `TU_PRIMARY` constant to prevent local reinterpretation.
- `visualisation/style.py`
  - imports the central palette;
  - uses `TU_PRIMARY` for the shared figure header.
- `visualisation/plot_sbx.py`
  - imports the central palette and assigns stable semantic roles;
  - removes the long figure-level current-status string that overlapped
    the first-row subplot titles;
  - reduces figure height, margins, row/column spacing, title size, and
    repeated legend footprint;
  - keeps all three corridor rows and the full-width cumulative-payment
    row while making the plot materially tighter.

## Verification

- Python compilation passed for `color_config.py`, `style.py`, and
  `plot_sbx.py`.
- The compact three-corridor layout completed a 15-minute headless smoke
  render before the final colour-role-only follow-up; the preview shows
  no figure-status/title collision.
- Current focused suite result: **46 passed, 1 failed**. The remaining
  failure is unrelated configuration drift: `005_CIGRE_MULTI.py` still
  passes the removed `enable_tie_coordination` argument to
  `MultiTSOConfig`.
- A fresh experiment-014 render is additionally blocked by an unrelated
  `IndentationError` in `experiments/runners/multi_tso_dso.py` around
  lines 366-367. These concurrent architecture changes were not altered.

## Open point

5c yellow-green has lower contrast than dark blue on a white background.
It remains appropriate for the primary line and header, but red/teal and
magenta are retained for categorical state and settlement semantics so
the plot does not encode every quantity with the primary colour.
