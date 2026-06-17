# 2026-06-17 — Drop ∂I/∂u rows from DSO H; scale Kalman process noise by ‖Δu‖²

## Reason
For the CIGRE 2026 estimation study (`experiments/003_S_DSO_CIGRE_2026.py`):

1. **No currents in H.** The paper's output vector is `y = [Q_int, V]` (chapter 2,
   `eq:sensitivity`), but the DSO_2 controller was monitoring HV line currents, so
   `_H_cache` carried `[Q_trafo | V | I_line]` rows. The Kalman filter flattens the
   *full* matrix, so `∂I/∂u` entered the estimator state, the innovation `Δy`, the
   (full, non-diagonal) measurement covariance `R`, and the exported Frobenius error.
   The `"q_trafo+v"` row-mask only blocked write-back, not the estimation/metric.
   Decision (with M. Schwenke): drop the current rows entirely (Option A), accepting
   removal of the line-current MIQP constraints (non-binding on the stiff slack-decoupled
   island).

2. **Δu²-gated process noise.** The KF random-walk process noise `Σ_q` was scaled by
   `s ∝ ‖Δu‖` (linear). Switched to `s² ∝ ‖Δu‖²` — the covariance interpretation
   (if the per-step H increment scales like Δu, its covariance scales like ‖Δu‖²).
   Stronger shut-off of fictitious diffusion at convergence (Δu→0), the main driver
   of the frozen-OP drift in the results chapter.

## What changed
**(1) Drop current rows — new flag `dso_monitor_currents: bool = True`**
- `configs/multi_tso_config.py` (`MultiTSOConfig`): new field; `True` = legacy
  (monitor currents, H has `∂I/∂u`, MIQP enforces current limits), `False` =
  `current_line_indices=[]`, H reduces to `[Q_interface | V]`, no current limits.
- `experiments/runners/multi_tso_dso.py` (~L760): `hv_lines` gated by the flag —
  `[]` when `dso_monitor_currents` is False. `hv_line_max` follows (empty ⇒ `None`).
  Default-True keeps 002/005/006 unchanged.
- `experiments/003_S_DSO_CIGRE_2026.py` (`make_config`): set
  `cfg.dso_monitor_currents = False`.

**(2) ‖Δu‖² scaling** in `_KalmanHPredictor.__call__`
(`experiments/003_S_DSO_CIGRE_2026.py`, predict step):
- was `s = sqrt(max(0, du_n2 - pe_dither))/sqrt(n)`; `P_p = P/λ + s·Σ_q`
- now `s2 = max(0, du_n2 - pe_dither)/n`; `P_p = P/λ + s2·Σ_q`
  (`s2` is literally the square of the old `s`). PE-dither-energy subtraction and
  trace cap unchanged.

## Verification
- `py_compile` on the three files: **NOT YET RUN** — the command-execution safety
  classifier was temporarily unavailable at edit time. Re-run before committing:
  `python -m py_compile configs/multi_tso_config.py experiments/runners/multi_tso_dso.py experiments/003_S_DSO_CIGRE_2026.py`

## Impact / TODO
- **MUST regenerate Kalman matrices.** Dropping the current rows changes `n_y`
  (hence `n_state = n_y·n_u`), so the on-disk `kalman_matrices.npz` (Q/R) no longer
  matches the filter's `C = I_{n_y} ⊗ Δuᵀ`. `_load_matrices` does not validate the
  on-disk shape. Re-run, in order, with the flag already off:
  `collect_training_data()` → `generate_kalman_matrices()` → `run()`.
  (Re-collect, not just re-slice: removing current constraints can change the
  control trajectory and hence the Q/R statistics.)
- The `‖Δu‖²` scaling alone needs no retraining (runtime multiply on Q).
- Paper sync: `3_estimation.tex` prediction-covariance eq. should become
  `Σ_{P,t|t-1} = Σ_{P,t-1|t-1}/λ_f + s_t²·Σ_Q`,
  `s_t² = max(0, ‖Δu_t/u_scale‖² − E_dither)/n_u`. Stale `5_conclusion.tex` draft
  `y=[Q,V,I]` → `[Q,V]`.
- `H_PREDICTOR_ROWS="q_trafo+v"` is now effectively a no-op mask (no current rows
  to keep from cache); `"all"` would behave identically.
