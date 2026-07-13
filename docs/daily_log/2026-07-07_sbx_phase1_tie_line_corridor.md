# 2026-07-07 — SBX Minimal Phase 1: tie-line + corridor model

**Task:** SBX build plan v2 (with v2.2 amendment), Phase 1.
**Changed (new files only, no existing module touched):**

- `sbx/__init__.py`, `sbx/fail.py` (`rep1()` fail-fast helper, A1/G1),
  `sbx/config.py` (frozen `SBXConfig`, plan §5 defaults, derived
  `t_cycle_min` / per-cycle `dq_quant_mvar`).
- `sbx/tie_line_model.py`: `TieLineParams` (pu on `net.sn_mva`/`vn_kv`,
  per-end half shunts), `extract_tie_line_params` (reads `net.f_hz` —
  60 Hz on IEEE 39), `q_flow` (δ via `brentq`, transfer-limit + bracket
  asserts), `v_sched_for_q` (nested `brentq`), `sensitivities` (total
  derivatives via central FD, step-halving consistency ≤ 1 %).
- `sbx/corridor.py`: corridor registry against the BME partition
  (line-only cross-area assert), `corridor_q_flow`, `corridor_solve_dv`
  (Step-4 common-`dv` root find), `corridor_sensitivities` (per-line +
  per-side sums for the v2.2 joint-box LP).
- `tests/sbx/test_golden_tie_line.py`: golden tests 1–4 + registry
  ground-truth + corridor-sum consistency.
- `STATUS_SBX.md`: Phase 0 gate resolution, v2.2 amendment recorded
  verbatim, Phase 1 section with golden-test results.

**Method:** symmetric π model, y = g + jb (b < 0); P/Q equations per plan
§3 with the g_sh term carried explicitly; δ implicit via the P-equation
everywhere (total, not partial, derivatives). Corridor root find always
evaluates q_corr at the reference end A; `dv` is applied to the acting
side (clarification vs the plan's Step-4 formula, recorded in
STATUS_SBX.md §1.1).

**Result:** 6/7 tests pass. Golden test 4 FAILS structurally: 1e−3 pu
contract-voltage rounding (plan §2.1) shifts q_std by +0.9 to +4.1 Mvar
(|b| ≈ 40–75 pu → 4–7.5 Mvar per 1e−3 pu of voltage-difference error),
beyond the max(0.5 Mvar, 1 %) golden tolerance on all three corridors.
Recommendation at the gate: round v_std to 1e−4 pu instead (deviation
scales ~×10 down). Options in STATUS_SBX.md §1.3.

**Reason:** Phase 1 is the physics foundation for contract data (Phase 2),
capability (Phase 3) and settlement attribution (Phase 6); the golden
tests pin extraction, signs, per-unit conversion and inverse solves before
any protocol logic exists.

**Status:** Paused at the Phase 1 gate (rounding decision pending).
