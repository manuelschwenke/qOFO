# 2026-04-14: Fix factor-of-2 bug in stability analysis (C1, C2, C3)

## What changed

### Bug: Curvature matrix used Hessian (2C) instead of curvature (C)

The MIQP first-order condition `w* = -grad_f / (2 G_w)` introduces a factor 1/2
that cancels the factor 2 in the Hessian `nabla^2 f = 2 H^T Q H`. The effective
iteration gain is `alpha * G_w^{-1} * C` where `C = H^T Q H` (no factor 2).

The code was building `Phi_c = 2R + 2 K^T Q K` and computing
`rho = |1 - alpha * lambda(G^{-1/2} Phi_c G^{-1/2})|`, which is 2x too large.

### Files modified

| File | Change |
|------|--------|
| `analysis/stability_analysis.py` | C1: `Phi_c = 2R + 2C` -> `C_c = R + C`; C2: same; C3: factor 4 -> 2 |
| `analysis/auto_tune.py` | `_build_M_cont`: removed factor 2; `_build_M_full_c`: removed factor 2; `tune_discrete_gw`: 4 -> 2 |
| `docs/stability_analysis.md` | Added derivation note explaining factor cancellation |
| `docs/stability_report_for_discussion.md` | Updated formulas to use C (not 2C), factor 2 (not 4) |

### Key method / structure

- `_spectral_radius_iteration` unchanged (formula `|1 - alpha*lambda|` is correct when M is built from C)
- Gershgorin threshold `C_kk / 2` (was `2C_kk / 2 = C_kk`) -- halves required g_w
- C3 Gamma: `(2/g) * row_L1` (was `(4/g) * row_L1`) -- halves required g_w for discrete

### Expected impact

- C1 rho drops from ~3.8-4.2 to ~1.2-1.6 (still needs alpha < 1 or modest g_w increase)
- C3 rho(Gamma) drops from ~1.08 to ~0.54 (passes with current g_w = 10)
- Auto-tuner will recommend alpha ~0.8 instead of ~0.4

### Tests

All 12 tests in `test_auto_tune.py` pass. Pre-existing failures in other test files unrelated.

## Reason

C1 always failed when tracking performance was set high (large g_q, low g_w),
making the stability analysis appear 2x more conservative than the actual system.
