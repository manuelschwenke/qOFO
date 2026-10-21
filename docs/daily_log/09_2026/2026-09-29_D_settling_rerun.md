# Experiment D rerun with a subordinate-settling limit (thesis Ch 8) -- part 1

Timestamp: 2026-09-29T06:10+02:00
Reason: handoff `HANDOFF_D_settling_rerun.md` (2026-09-28). The selected weights
`399e30d3b60b` settle the STS loops too slowly for `N_inner = 9`
(`n_settle` pooled p95 23). Author decision: keep `T_STS = 20 s`, `N_inner = 9`,
re-select the weights so that pooled p95(`n_settle`) <= 9.

## Code changes

### `experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py`

- **New metric `n_settle_final_1mvar`** (`n_settle_final()`): first iteration
  from which the interface flow stays within 1 Mvar of its FINAL value; censored
  at the observation length (46) if the flow moves > 1 Mvar in the last 10
  iterations. Exactly the handoff's reference rule. New columns
  `n_settle_final_1mvar`, `n_settle_censored` in `steps.csv`; all existing
  columns kept.
- **`aggregate_settle()`** -> `n_settle_summary.csv` and a new first section in
  `summary.md`: per DSO and pooled median / p90 / p95 / max / share <= 9 /
  censored, residual `|q_final - q_set|` median / p95 / max / count > 1 Mvar,
  voltage violations, tap moves.
- **`--weights-json`**: complete weight vector applied with
  `tuning.parameters.apply_to_config` on top of `--paramset` (the same
  construction as the Stage-II screen); fingerprint recorded as candidate key.
  Job fields `weights` and `cfg_overrides` (existing `MultiTSOConfig` fields,
  e.g. `dso_g_q_per_area` for a diagnostic) are honoured by `_run_case`.
- Self-test extended (5 cases for the new metric); all pass.
- **Check against the archive:** applied to
  `results/THESIS_ch9_1_ninner/E_final_ladder/20260915-213349/trajectories.csv`
  it reproduces the handoff numbers exactly: pooled median 11 / p95 23.05 /
  max 33 / 42 % <= 9; DSO_1 13.5/20, DSO_2 9/14.65, DSO_3 19/25.55,
  DSO_4 7.5/11.55. No step censored.

### `tuning/settling_predictor.py` (new)

Linear prediction of `n_settle` for the isolated-STS protocol, used to pre-rank
DSO weight sets. Model: the STS continuous update is the gradient step
`w = -G_w^{-1}(g_q H_Q^T e_q + dso_g_v H_V^T e_v)` (confirmed in
`optimisation/miqp_solver.py`: objective `w^T G_w w + grad_f^T w`), iterated on
the captured DER columns of `H` (`_expand_H_to_der_level`), 10 Mvar step on one
interface, siblings held, scored with the same `n_settle_final`. Ignores
integer actuators, capability/voltage constraints, noise, nonlinearity.
Sensitivities cached per condition in
`results/stage1/operating_points/hsts_rural_700_net190b46e8_W*.pkl` (~19 s each).

### `experiments/ch_9_parameter_selection/ch_8_D1_settling_screen.py` (new)

D1 driver: runs the unchanged step protocol (120 steps per set, W1-W4 + W6,
frozen OFO parent, 10 Mvar, trajectories saved) for every set in a JSON list,
all sets in one resumable process pool (one JSON per case), writes per-set run
folders in the step script's format plus `d1_screen.csv` (weights, analytic
`lambda_max(B)` and binding layer, linear prediction, measured pooled and
per-DSO p50/p95/max, residuals, violations, tap moves).

## Findings so far (analytic, before simulation)

Assumptions: DSO_3 x2 network, `make_config()` base, TSO part of `399e30d3b60b`,
`dso_gamma_oltc_q = 0`, g_w adapters off (read from the E run's
`resolved_config.json`).

1. **Predictor calibration at `399e30d3b60b`.** W2 and W6: predicted per-interface
   `n_settle` matches the simulation almost exactly (e.g. DSO_3|trafo_6 19 vs
   19/19). W1, W3, W4: simulation 4-10 iterations slower (constraints active
   there, hypothesis). Pooled p95 predicted 19 vs measured 23. The predictor is
   **optimistic** (a lower bound in practice), and it reproduces the per-DSO
   ordering (DSO_4 < DSO_2 < DSO_1 < DSO_3).
2. **`dso_gamma_oltc_q = 0`**: DSO taps carry no interface-Q gradient, so
   `g_w_dso_oltc` cannot accelerate interface-Q settling (E run: 0 DSO tap moves
   in 120 steps).
3. **`dso_g_v` is practically inert for the settling** over three decades
   (voltage curvature small against the interface-Q curvature at these weights).
4. **The settling speed is set by `g_q / g_w_dso_der` and the conditioning of the
   DER->interface-Q map** `H_Q H_Q^T` (W1): cond 6.8 / 5.6 / **9.5** / 5.2 for
   DSO_1..4. The slow mode is the differential mode between the three parallel
   interface transformers of one HV network, which DER reactive power barely
   distinguishes. A uniform DER weight cannot speed that mode up without pushing
   the common mode past `lambda_max(B) <= 1.6`.
5. **Global ceiling**: `g_q` x1.286 on `399e30d3b60b` reaches
   `lambda_max(B) = 1.6` at STS_DSO_4@W6. Predicted pooled p95 **15** (DSO_3 15),
   i.e. no global DSO weight set is predicted to reach p95 <= 9, even by the
   optimistic model.
6. **Per-area `g_q`** (each area at its own STS ceiling; an architectural change,
   run as a diagnostic only): DSO_1/2/4 predicted p95 <= 8, DSO_3 still **11**.
7. **Loophole in the final-value criterion**: a set whose flow barely responds
   (g_q ~ 0.002) "settles" at iteration 0 with a residual of ~10 Mvar. 150 of the
   205 lambda-feasible Sobol draws are of this non-tracking kind in the linear
   model. The criterion is kept as specified; the selection of Sobol draws to
   simulate used "predicted residual median <= 1 Mvar" as a sampling heuristic
   only. **Open question for the author.**

## Runs started 2026-09-29 ~06:05

- D1: `results/THESIS_ch8_D_settling_20260929/d1/` (sets from
  `d1/make_sets.py`): `glob_ceil`, `perarea_ceil_DIAG`, `ref_399e30d3b60b`,
  `glob_ceil_lowv`, `glob_ceil_lowoltc`, and 6 fastest-predicted tracking Sobol
  draws (seed 20260929; 512 draws, 205 pass lambda, 55 predicted to track).
  30 workers.
- C: `results/THESIS_ch8_D_settling_20260929/C_initial_settling/` --
  `--paramset stage1_start_dso3x2 --max-step-mvar 10 --save-trajectories`,
  6 workers.

## Cost profile of one step case (cProfile, W1 / DSO_3|trafo_6 / up, `ch9_selected`)

- Result reproduces E for this case: `n_settle = 19` (E: 19), residual 0.62 Mvar.
- 2227 s wall under the profiler (~964 s unprofiled median in E). ~96 % is the
  plant: `core/plant.py::advance` -> pandapower `run_control` with ~45 power
  flows per record (DER local Q(V) control loop), dominated by pandas overhead
  in `_pd2ppc`. The MIQP solves are minor. Not changed: speeding up the plant
  loop would touch the physics of every campaign.

## Results, 2026-09-29 ~13:00 (simulated, step protocol, 120 steps each)

| set | pooled p50 / p95 / max | share <= 9 | DSO_1 p95 | DSO_2 p95 | DSO_3 p50 / p95 | DSO_4 p95 | residual med [Mvar] | V viol | DSO taps |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| C `stage1_start_dso3x2` | 25 / 32 / 34 | 10 % | 31.6 | 29.6 | 27 / 33 | 29.7 | 2.27 (92/120 > 1) | 0 | 0 |
| E `399e30d3b60b` (archive) | 11 / 23.05 / 33 | 42 % | 20 | 14.65 | 19 / 25.55 | 11.55 | 0.39 | 0 | 0 |
| `glob_ceil` (g_q x1.28, lambda 1.592) | 9 / 20.05 / 33 | 55 % | 17 | 13.55 | 15 / 23 | 10.1 | 0.29 | 0 | 0 |
| `perarea_ceil_DIAG` (not selectable) | 7 / 16.05 / 32 | 72 % | 14 | 13.55 | 12 / 20 | 10.1 | 0.25 | 0 | 0 |

Linear prediction for `glob_ceil`: p95 15 (measured 20.05); for the per-area
diagnostic 11 (measured 16.05) -- optimistic by ~5 iterations, as calibrated.
DSO_3 median by window at `glob_ceil`: W1 17, W2 12, W3 18, W4 19.5, W6 12.5.

**Conclusion (established for these two points, consistent with the analytic
screen): the best global DSO weight set misses p95 <= 9 by a factor ~2
(20.05), and even per-area g_q misses it (16.05, DSO_3 p95 20).** This is the
handoff section-4 stop condition. D2 and E are NOT started; remaining D1 sets
(reference, two ceiling variants, 6 Sobol) continue for the trade-off record.

## D1 complete, 2026-09-29 ~18:20 (12.3 h, 1320 cases, 0 failures)

| set | g_q/g_w_dso_der | lambda | pooled p95 | DSO_3 p95 | residual med | DSO taps |
|---|--:|--:|--:|--:|--:|--:|
| ref_399e30d3b60b | 0.256 | 1.310 | 23.05 (= E archive) | 25.55 | 0.39 | 0 |
| glob_ceil | 0.328 | 1.592 | 20.05 | 23.00 | 0.29 | 0 |
| glob_ceil_lowv (dso_g_v x0.1) | 0.350 | 1.592 | **19.05** | 22.00 | 0.21 | 0 |
| glob_ceil_lowoltc (g_w_dso_oltc x0.316) | 0.299 | 1.592 | 26.05 (1 censored) | 30.55 | 0.23 | 30 |
| sobol_00..05 | 0.27-0.34 | 1.40-1.56 | 19.05-21.05 | 22.6-25.7 | 0.22-0.42 | 0 |
| perarea_ceil_DIAG | per area | 1.592 | 16.05 | 20.00 | 0.25 | 0 |

- The reference reproduces the E archive exactly (pooled p95 23.05): the
  protocol is deterministic across the code changes since 47c6b70.
- Cheaper DSO taps engage (voltage-driven, 30 moves) and make settling WORSE.
- The linear predictor ranks the sets with r = 0.72 (p95) and is optimistic by
  ~4-5 iterations.
- 0 of 11 sets reach p95 <= 9 -> handoff section-4 stop; D2/E not started.
- Partial bundle README: `results/THESIS_ch8_D_settling_20260929/README.md`.
