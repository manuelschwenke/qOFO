# 2026-06-29 - Mutual-gradient coordination notebook

**Timestamp:** 2026-06-29 15:15:36 +02:00  
**Scope:** Added an experiment notebook to demonstrate the horizontal TSO-TSO
mutual-gradient coordinator across `tie_grad_eps` and `tie_grad_step`.

## Change

- Added `experiments/008_TIE_MUTUAL_GRADIENT_DEMO.ipynb`.
- The notebook reuses the validated divergent-schedule setup from
  `experiments/007_TIE_COORDINATION.py`.
- It compares `enable_tie_coordination=False` against a grid of coordinated
  cases with varied `tie_grad_eps` and `tie_grad_step`.
- It reports steady-window mean zone V-RMS error from reference, total
  `sum |Q_tie|`, negotiated `dV_ref`, and combined boundary-gradient diagnostics.
- When executed, it caches summary results and figures under
  `results/008_tie_mutual_gradient_demo/`.

## Method / Structure

1. Locate the project root from the active notebook working directory.
2. Import `007_TIE_COORDINATION.py` via `importlib`, because the module name
   starts with a digit.
3. Build each run from `_base_config(live=False)` and apply the existing
   `TIE_KW` coordination defaults.
4. Run one uncoordinated baseline and a parameter grid over
   `tie_grad_eps x tie_grad_step`.
5. Compute steady-window metrics over the last 30 minutes:
   mean zone V-RMS error in mpu, summed absolute tie-line reactive exchange,
   mean absolute `dV_ref`, and mean absolute combined gradient.
6. Plot heatmaps, Pareto-style voltage/tie-flow views, and one diagnostic trace
   showing voltage tracking, tie exchange, combined gradients, and negotiated
   boundary shifts.

## Reason

The existing `007_TIE_COORDINATION.py` already validates the divergent-schedule
coordination case and sweeps `tie_grad_eps`. The new notebook makes the mutual
gradient mechanism easier to inspect interactively and extends the empirical
view to `tie_grad_step`, so the tuning surface can be discussed in terms of
voltage-reference tracking rather than only tie-flow reduction.

## Notes / Risks

- The notebook is not pre-executed; running the full default grid performs 31
  simulations and may take time on the network workspace.
- Cached CSV summaries are used only for aggregate plots. The diagnostic trace
  reruns the selected case if raw logs are not already present in memory.
- The result remains scenario-specific: IEEE-39, divergent voltage schedules,
  current cached-sensitivity/controller setup, and the CIGRE actuator set.

## Refinement update

**Timestamp:** 2026-06-29 16:48:58 +02:00

**Change:** Updated experiments/008_TIE_MUTUAL_GRADIENT_DEMO.ipynb after the first sweep results.

**Method / Structure:**

- Confirmed from esults/008_tie_mutual_gradient_demo/mutual_gradient_sweep.csv that the best first-pass point is 	ie_grad_eps=1e-4, 	ie_grad_step=1.0 by steady-window mean zone V-RMS error.
- Replaced the sweep grid with a local refinement around that point: 	ie_grad_eps=[5e-5, 7.5e-5, 1e-4, 1.25e-4, 1.5e-4, 2e-4, 3e-4] and 	ie_grad_step=[0.75, 0.9, 1.0, 1.1, 1.25, 1.5].
- Added SWEEP_TAG=refined_eps1e-4_step1 so the refined CSV and figures do not overwrite the first-pass outputs.
- Changed the diagnostic gradient and dV_ref plots to use point markers, because those fields are sparse coordination-update records and line plots through NaN gaps can appear empty.

**Reason:** Prepare a denser empirical check around the apparent tuning optimum and make the mutual-gradient diagnostics visible in the notebook output.

**Risk / Note:** The refined grid has 42 coordinated runs plus the OFF baseline. The diagnostic bottom panels show update-sample points; a later revision could add forward-filled step plots for state-like dV_ref if continuous visualization is desired.
