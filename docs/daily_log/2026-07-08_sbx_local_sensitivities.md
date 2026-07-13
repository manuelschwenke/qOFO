# 2026-07-08 — SBX: local-sensitivities restriction lifted

**Task:** Manuel: "I want SBX to work with the local sensitivities."

**Analysis:** the Phase-5 runner validation excluded
`local_sensitivities_*` under `coordination_mode="sbx"` as a
conservative fail-fast on an unvalidated path. Structurally nothing in
SBX requires the shared Jacobian: `sbx/adapter.py` reads only
controller-owned cached objects (voltage rows of the controller's own
H via `_build_sensitivity_matrix`/`_expand_H_to_der_level`;
`ctrl.sensitivities.compute_dV_dQ_der` for the relieving-sign assert),
and the local mode builds the same `JacobianSensitivities` class over
the Ward-style reduced zone net, which retains all in-zone buses
(corridor terminals, violated buses) under original indices. Local
cached models are the configuration most consistent with the SBX
locality principle (plan §2.4 "local data only").

**Changed:**
- `experiments/runners/multi_tso_dso.py`: sbx validation now excludes
  only `numerical_h` (no analytic dV/dQ at all); explanatory comment.
- `experiments/014_SBX_SINGLE_DEMO.py`: `--local-sens` flag
  (TSO + DSO local models) and `--arm sbx|sbx_inert|none` (supports
  baseline probing without editing the file; arm "none" exits cleanly
  without SBX outputs).

**Validation:** asym_z3, 150 min, headless, `--local-sens`: exit 0;
full protocol arc (deals c3–c4 on (1,3) and (2,3), unwind c7–c8 to
zero, refs back at v_std); dv per quantum identical to the shared-path
run (+1.68/+0.80 mpu); relieving-sign assert silent; settlement
completed. `sbx/` untouched.

**Note:** 013's campaign keeps the shared path (arms already run;
comparability). A local-vs-shared SBX comparison would be a cheap
follow-up experiment if wanted.
