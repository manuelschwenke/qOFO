# 2026-07-07 — SBX Phase 1 gate resolution + Phase 2: contract data

**Task:** SBX plan v2/v2.2, Phase 1 gate closure and Phase 2.

**Gate resolution (contract-voltage rounding):** Manuel chose "round to
1e-4 first". Measured: 1e−4 pu passes corridors (1,2)/(2,3) but fails
(1,3) marginally (−0.562 vs 0.500 Mvar tol; |b| ≈ 58 pu → worst case
0.58 Mvar per 1e−4 pu, so 1e−4 cannot guarantee the 0.5-Mvar golden
floor). Escalated per the "first" phrasing: **1e−5 pu adopted** (worst
deviation 0.067 Mvar, ≥ 85 % margin everywhere). Normative constant
`sbx/contract.py::V_STD_DECIMALS = 5`, decision trail in its docstring.

**Changed:**

- `sbx/contract.py` (new): frozen `CorridorContract`,
  `build_default_contract` (base-case v_std, config constants, converged-
  result guard), `q_std_mvar` (Step-1 pure function), rate-scaled
  `dq_quant_mvar`.
- `core/measurement.py` (approved G2 interface extension, additive):
  optional `Measurement.tie_line_p_mw` + population in
  `measure_zone_tso` (`p_from_mw`/`p_to_mw` at the in-zone endpoint,
  load convention, aligned with the existing tie-line Q block).
- `tests/sbx/test_golden_tie_line.py`: golden test 4 now at 1e−5 pu via
  `V_STD_DECIMALS` (history documented in the module docstring).
- `tests/sbx/test_contract.py` (new): Phase 2 acceptance tests.
- `STATUS_SBX.md`: §1.4 gate resolution + Phase 2 section.

**Result:** `pytest tests/sbx` 14 passed; regression on measurement-
touching suites (`test_measurement`, `test_controller`,
`test_tie_coordination_hooks`, `test_tso_loss_objective`) 71 passed /
1 pre-existing skip.

**Reason:** Phase 2 freezes the bilateral contract data (v_std, band,
quantum rate, caps, prices) and provides the deterministic Step-1
standard both areas evaluate identically; the measurement extension
supplies the per-line P that the standard needs (persistence forecast).

**Status:** Phase 2 ✅. Continuing with Phase 3 (need flag + v2.2
joint-box capability LP).
