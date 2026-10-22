# 2026-07-20 — Jacobian warm-start guard and shunt-update identity

**Timestamp:** 2026-07-20 17:37 CEST
**Reason:** resume the interrupted Phase 6 work and resolve the reported
`NoneType.todense` failure in `JacobianSensitivities.__init__` without
reverting the uncommitted 2026-07-17 warm-start work.

## Established failure mechanism

Pandapower may accept `init="results"` without executing a Newton-Raphson
iteration when the starting point already meets the mismatch tolerance. In
that case the load flow is converged but
`net._ppc["internal"]["J"]` remains `None`. The former `1e-8` perturbation
was applied to the first pandapower bus, which is commonly the slack bus and
therefore not part of the Newton state vector; it did not guarantee an
iteration on the small three-bus DSO test.

## Revision

- Added `runpp_with_stored_jacobian` and private helpers in
  `sensitivity/jacobian.py`.
- The first retry perturbs an actual PQ-bus voltage magnitude by `1e-6 pu`;
  if there is no PQ bus it perturbs a PV-bus angle by `1e-4 degree`.
- A failed results start retains the existing flat-start convergence
  fallback. If the solve converges but still stores no Jacobian, a stronger
  state kick (`1e-4 pu` or `1e-2 degree`) and `tolerance_mva <= 1e-12` force
  a final results-start iteration. Absence of `J` after that is an explicit
  error rather than a later `None.todense` exception.
- `sensitivity/network_reduction.py::build_tso_local_net` now uses the same
  helper, eliminating the duplicate guard with the same slack-bus hole.
- `SensitivityUpdater` now snaps a shunt `V^2` ratio to exact identity when
  it is within `1e-10` relative of one. Repeated NR solves shifted the test
  voltage by only `2.59e-11 pu`; treating that numerical repeatability noise
  as a physical shunt-state change modified one H column by up to
  `7.20e-12`. The physical state-dependent rescaling remains unchanged for
  resolvable voltage changes.

## Assumptions and constraints

- The sensitivity implementation requires pandapower's Newton-Raphson
  Jacobian with its standard `[P_PV, P_PQ, Q_PQ]` structure; distributed
  slack is disabled for this internal sensitivity solve.
- The perturbations affect only the NR initial guess and are removed by the
  converged solution.
- The controller hierarchy still sees only cached sensitivities and
  measurements. No plant state or RMS equation is exposed to either TSO or
  DSO controller.
- This change does not alter the actuator set (DER Q, AVR references,
  OLTCs, MSC/MSR) or the controlled outputs (interface Q, nodal/zone
  voltages and current constraints).

## Verification

- Focused sensitivity suite:
  `tests/test_sensitivity_updater.py tests/test_dso_qv_sensitivity.py` —
  **14 passed**.
- Broader Phase 1–6 regression set (Jacobian/Q(V), ZIP load model,
  auxiliary buses, plant abstraction, replay analysis, PF naming and
  snapshot round-trip) — **77 passed, 6 skipped**. The skipped tests are
  explicitly opt-in/live-PowerFactory adapter checks.
- Updated Python modules compile successfully.

## Risks and unresolved points

- The helper is deliberately NR-specific. A future pandapower algorithm
  that does not populate `internal.J` must not be routed through it.
- `JacobianSensitivities` still densifies the sparse Jacobian; this is a
  pre-existing scalability question and was not changed here.
- The `1e-10` identity threshold is numerical, not a physical voltage
  deadband. It is approximately nine orders of magnitude below the
  configured Q(V) deadband and does not replace measurement-noise handling.
