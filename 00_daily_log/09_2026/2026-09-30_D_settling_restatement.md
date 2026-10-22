# Experiment D: relax lambda_max(B) or restate the settling requirement? (analysis, no code change)

Timestamp: 2026-09-30. Context: `2026-09-29_D_settling_rerun.md` (0 of 11 D1 sets reach
pooled p95(n_settle) <= 9). Author asked whether to relax lambda_max(B) or restate
the requirement, and whether cascade time separation must assume a settled inner layer.
Script: scratch `restate.py` (not kept in repo); inputs = D1 per-set folders, C run,
cached STS sensitivities.

## Floor of a scalar-weighted gradient STS loop (established, linear model)
Best contraction of a gradient iteration with one scalar step on a subspace with
condition number kappa is (kappa-1)/(kappa+1), at lambda_max = 2 kappa/(kappa+1).
kappa = cond(H_Q H_Q^T), W1-W4+W6: DSO_1 6.6-7.0, DSO_2 5.4-5.8, **DSO_3 9.3-9.7**,
DSO_4 4.8-5.6 -> best factor for DSO_3 0.81 -> >= 10.7-11.2 iterations for a 10:1
reduction in the worst direction, **independent of the lambda cap**.

## Relaxing lambda_max(B) (linear prediction, optimistic by ~4-5 iterations)
cap 1.60 -> pred pooled p95 15; 1.70 -> 14; 1.80 -> 13; 1.90 -> 12; 1.97 -> 12.
Relaxation buys ~3 iterations and spends the 20 % margin against sensitivity mismatch.

## Delivered fraction within one supervisory period (k = 9), simulated
r9 = |q(9)-q_final| / |q_final-q_before| (denominator floored at 1 Mvar).
p50 / p95 of r9: best global sets 0.10-0.13 / 0.39-0.42; 399e30d3b60b 0.154 / 0.496;
per-area DIAG 0.075 / 0.486; glob_ceil_lowoltc 0.12 / 0.67; C start 0.40 / 0.87.

## Composite model (hypothesis-level derivation, one interface-Q mode, scalar)
TSO: s_{j+1} = s_j - k (q_j - q*), k < 2 its own OFO gain on the mode (model assumes delivery).
STS: q_{j+1} = q_j + delta (s_{j+1} - q_j), delta = 1 - r9.
z^2 - (2 - delta(1+k)) z + (1 - delta) = 0; Jury: stable iff 0 < delta < 2 and
k < 4/delta - 2, i.e. stable for every delta in (0,1] whenever k < 2. With complex
poles |z| = sqrt(1-delta). -> delta governs damping/convergence of the cascade, not
stability (in this model). Candidate restatement: delta_min = 1 - rho_c^2 at p95.

## Simple rule proposed to the author (2026-09-30)
Criterion: p95 over the 120-step bank of `n_settle_final_1mvar` <= 3 N_inner = 27
(settled within three supervisory periods, 9 min); side condition: median residual
|q_final - q_set| <= 1 Mvar (closes the non-tracking loophole).
- 399e30d3b60b (E archive = D1 rerun, identical): p95 23.05, per DSO 20 / 14.65 /
  25.55 / 11.55, 3 of 120 steps > 27 (max 33); median residual 0.39 Mvar -> PASS.
  Median delivered fraction at k = 9: 0.85.
- stage1_start_dso3x2 (C): p95 32, 36 of 120 steps > 27, median residual 2.27 Mvar -> FAIL.
- Two periods (18) would fail for 399e (22 steps > 18).
- Rejected alternative: "time constant (63 %) <= T_TS" at p95 -- n_63 p95 = 17.45 for
  399e (median 4), so a literal time-constant criterion at p95 fails.

## Author decision and bundle finalised (2026-09-30)
- Criterion adopted: p95(n_settle) <= 3 N_inner = 27 (dominant STS time constant
  <= T_TS, request settles within three time constants), evaluated on the p95 over
  the 120-step bank; side condition median residual <= 1 Mvar.
- `ch9_selected` = 399e30d3b60b unchanged; D2 not rerun.
- Bundle `results/THESIS_ch8_D_settling_20260929/`: README rewritten,
  `E_final_settling/` (= D1 ref set, identical to the E archive),
  `criterion_check.py/.txt`, `criterion_thesis_text.tex` (straightforward wording
  per author request), `weights_table.tex` (config names in column 1).
- Noted in README: all 11 D1 sets pass the restated criterion, so it separates the
  selected region from the starting weights but does not discriminate among
  near-selected sets.
