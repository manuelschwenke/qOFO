# Tap engagement rule: remove the ||a_i||² term (Stage-I screen)

- Timestamp: 2026-10-14 (CEST).
- Reason: While stating the tap engagement rule in the thesis (Ch 4 `eq:ofo_tap_engagement`), the rule in `tuning/stage1_screen.py` turned out not to match the controllers' MIQP. The MIQP step objective is `w^T G_w w + c^T w + z^T G_z z` with diagonal `G_w` and `g_u = 0` (`optimisation/miqp_solver.py`); with no output limit active it splits per actuator, so a free tap commits one step iff `|c_i| > g_w_i` (its step is the relaxed step `-c_i/(2 g_w_i)` rounded). The screen used `2|a_i^T z| > ||a_i||² + g_w_i`, i.e. the condition for an objective with an additional curvature term `w^T H^T Q H w`, which the MIQP does not have.

## What changed

| File | Change |
|---|---|
| `tuning/stage1_screen.py` | Module docstring: commit rule and both formulas corrected (`Δv_eng = g_w / (2 g_V |Σ H_V|)`, `g_w = 2 g_V |Σ H_V| Δv_eng`), with a CORRECTED note. `engage_deviation`: numerator `g_w` (was `g_w + ||a_i||²`). `g_w_for_engage`: `2 g_V |h| d_engage` (was `… − ||a_i||²`); now returns 0 only for a column without voltage effect in the chosen reading. `design_integer_weights`: docstring and error message updated accordingly. `g_q` stays in the signatures (callers), but no longer enters. `LayerCurvature.column_norm_sq` kept (now unused here). |
| `tests/test_tap_engagement_rule.py` | New, 16 tests: the controllers' `MIQPSolver` commits a tap iff `|c| > g_w`, tap step = rounded relaxed step, continuous step independent of the tap; `engage_deviation` / `g_w_for_engage` inverse to each other without `||a_i||²`; column without voltage effect → not achievable. |

## Validation

- `tests/test_tap_engagement_rule.py`: 16 passed. No other test imports `stage1_screen`.
- Toy check before the fix (scratch script, `MIQPSolver`, `g_w_tap = 1`, `||a||² = 4`): tap moved for `|c| ≥ 1.05`, the old rule predicted no move up to `|c| = 5`.

## Consequences / open

- **The Stage-I starting tap weights (`stage1_start`, computed before this fix) are smaller by `||a_i||²`** than the stated `d_engage` (1 % TS, 2 % STS) requires, so at the starting point the taps engage earlier than stated. The final weights (`ch9_selected`) were selected empirically around that start (Experiment B), so the thesis results are not affected; only the description of the starting point is. No re-run.
- The stored paramsets in `configs/paramsets.py` are numbers and unchanged. Re-running Stage I would give larger tap weights.
- The note in the module docstring "measured 2026-08-14, TSO taps engage at 0.98–1.31 % under a systematic offset" is kept as recorded; it is not reconciled with the corrected formula here.
- Not committed to git.
