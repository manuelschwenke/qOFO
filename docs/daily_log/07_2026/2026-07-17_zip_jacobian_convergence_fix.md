# 2026-07-17 (4th entry) — ZIP regression: JacobianSensitivities convergence fix

**Symptom.** `run_multi_system_ofo.py` crashed at runner init with
`LoadflowNotConverged ... after 10 iterations` from
`JacobianSensitivities.__init__` (`sensitivity/jacobian.py:197`) while
building the per-zone reduced TSO Jacobians
(`_build_tso_local_jac` → `build_tso_local_net`). First run under the new
anchored ZIP load model.

**Root cause.** The class's internal re-solve (needed to drop
distributed_slack so the Jacobian has the standard [P_PV, P_PQ, Q_PQ]
structure) used pandapower defaults: cold `init='auto'` (dc) start and the
default 10 NR iterations. Under constant-PQ the reduced per-zone nets
happened to converge inside 10 iterations from a cold start; under the
anchored ZIP model they need more. The incoming net is *always* converged
(the class asserts it), so a cold restart was both fragile and semantically
wrong — the purpose is the Jacobian *at the given operating point*.

**Fix** (`sensitivity/jacobian.py`): warm-start the re-solve with
`init="results", max_iteration=50`, preceded by the 1e-8 voltage kick so NR
performs ≥ 1 iteration and stores `_ppc['internal']['J']` (results-init can
otherwise converge in 0 steps — the reduced nets from
`build_tso_local_net` arrive already at the exact no-distributed-slack
solution). Flat-start fallback with 200 iterations. This mirrors the
established pattern in `sensitivity/network_reduction.py` (step 10 of
`build_tso_local_net`), which had solved the identical problem for its own
power flow.

**Verification.** Short end-to-end run (`make_config()`, `n_total_s=60`):
init passes the previously failing line, 3 plant steps complete with TSO +
DSO OFO dispatches and plausible Q-tracking under the ZIP plant.

**Separate finding — Gurobi licence.** On this (PF) machine the first MIQP
solve raises `GurobiError: HostID mismatch (licensed to a2c4b5de, hostid is
3a6f9b86)` — the licence is node-locked to the other machine.
`MIQP_SOLVERS = ['GUROBI']` is hard-coded (`optimisation/miqp_solver.py`,
deliberate performance choice), so experiments on this host currently stop
at the first MIQP. Options (user decision, not changed today):
(a) `grbgetkey` an academic licence for this host — recommended, keeps
solver performance identical across machines; (b) extend `MIQP_SOLVERS`
with `'SCIP'` as fallback — changes solve times, mixes solvers across
machines. Verification above used a runtime-patched SCIP solver list
(throwaway, not committed).
