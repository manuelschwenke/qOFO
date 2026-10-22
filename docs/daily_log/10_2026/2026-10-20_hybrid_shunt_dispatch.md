# Hybrid shunt dispatch

Timestamp: 2026-10-20, Europe/Berlin. Implemented and tested during the afternoon.

Reason: the author requested a normal OFO bank decision for large present
demand, plus independent slow pressure from persistent STS tracking error and
low reserve. The author selected reserve action near capability limits.

Changes:

- Added an optional `HybridShuntSolver` that appends integer bank variables to
  each TS MIQP. `g_w_tso_shunt` prices per-level movement.
- Added two independent per-bank demand memories. Slow candidates reoptimise
  ordinary controls and transfer the predicted bank contribution out of the
  delegated STS request while holding the total interface reference fixed.
- Kept one physical commit path with shared dwell, budget and level limits.
  Added a one-bank-step-per-zone constraint and hard predicted voltage checks.
- Included additional bank effects in public controller predictions and added
  per-zone memory/decision diagnostics to experiment records.
- Added configuration checks, a diagnostic AC comparison script and tests.
- Made Stage-I capture/cache loading reject hybrid mode because the current
  screen omits its added bank columns. Existing campaign defaults are unchanged.

Verification: 198 tests passed, one pre-existing skip. SCIP and Gurobi both
pass targeted support tests. AC directional tests cover engagement and release
of MSC and MSR banks. The full diagnostic results, assumptions, limitations and
tuning consequences are recorded in
`docs/tuning/2026-10-20_hybrid_shunt_design_and_tests.md`.

The CIGRE paper supports deadband integration and integral reset after
switching. The STS reserve extension remains a separately tested design.
The trial configuration is not a new selected parameter set.
