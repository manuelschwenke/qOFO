# 2026-07-07 — SBX Phase 5: scheduler, runner integration, closed-loop smoke test

**Task:** SBX plan v2/v2.2, Phase 5 (six-step cycle scheduler + control
integration + closed-loop smoke test).  Continues the session that
delivered `sbx/scheduler.py` and `tests/sbx/test_scheduler.py` (45
protocol-level tests green before integration work started).

**Changed:**

- `sbx/scheduler.py` (edited): `CorridorCycleRecord.q_meas_mvar` — the
  cycle-averaged measured corridor Q of the elapsed cycle (settlement
  quantity §2.5, `q_sched` vs `q_meas` plot series) was averaged but
  discarded; now stored per cycle.  New public accessor
  `SBXScheduler.last_need(area)` so the runner adapter computes the
  relieving-sign scalar for the SAME worst-violated bus the scheduler
  asserts against.
- `sbx/adapter.py` (new): `SBXRunnerAdapter` — maps runner objects
  (per-zone `Measurement`, `TSOController`) onto the plant-agnostic
  scheduler.  Per TSO tick: cycle protocol on boundaries (AreaCycleData
  from the controller's CACHED model only), then `record_step`;
  references written through the existing `update_voltage_setpoints`
  path (w_track ≡ g_v, A4/G5).  Documented conservative capability-box
  adjustments: integer actuators frozen in the joint-box LP; Q_PCC,set
  box anchored at measured interface Q + vertical-CAIR capability
  interval; box widened to contain the current u.  Records
  `terminal_v_history` per TSO tick (v2.2 item-3 margin check, Phase 7
  plots).  Joint-box LP backend pinned to HiGHS through the EXISTING
  `MIQPSolver(solver=...)` argument — the OSQP default stalls at
  `user_limit` on the LP's mixed column scales (Mvar vs pu); no solver-
  wrapper modification.
- `configs/multi_tso_config.py` (edited): `coordination_mode` docstring
  extended with `"sbx"`; new field `sbx_config: Optional[object]`
  (an `sbx.config.SBXConfig`; None → plan-§5 defaults with the run's
  `tso_period_s`).  Typed loosely so the config module does not import
  `sbx`.
- `experiments/runners/multi_tso_dso.py` (edited): `"sbx"` accepted by
  the mode validation with fail-fast cross-checks (mutually exclusive
  with `enable_tie_coordination`; requires the full-network analytic
  shared-Jacobian path; no single-zone partition); adapter constructed
  AFTER the final pre-loop power flow (contract defaults from the
  converged experiment base case, A7); one `on_tso_step` call in the
  TSO block after the zone measurements are built (before the zones
  solve); `sbx_adapter` exposed via the `pre_loop_hook` state dict
  (011/012 capture pattern).  BME modules, vertical CAIR, MIQP assembly
  and solver wrapper untouched.
- `tests/sbx/smoke_sbx_closed_loop.py` (new): two-arm closed-loop smoke
  test (none vs sbx, identical scenario: 005 config, Q-load contingency
  at a zone-3 bus, tightened zone-3 v_min), criteria C1–C7 per plan
  Phase 5 + v2.2 items 3/5.  Script, not pytest-collected (long run).

**Key structure:** runner → `SBXRunnerAdapter.on_tso_step(sbx_it,
measurements, controllers)` → `SBXScheduler.run_cycle` on boundaries /
`record_step` every tick → references back through
`update_voltage_setpoints`.  Both areas of a corridor simulated
in-process (A9); Step-3 determinism still exercised via double
evaluation + checksum.

**Later the same day (Manuel's three amendments + smoke iterations):**

- Amendment 1: border-actuator diagnostic
  (`sbx/adapter.py::_border_actuator_diagnostic`, hop 0/1 via 2W/3W
  windings) — found two zone-1 AVRs one trafo from corridor (1,2)
  terminals.
- Amendment 2: smoke-test reported numbers R1/R2/R3 + criterion C8
  (supporter areas violation-free — the joint-box guarantee under
  test).
- Amendment 3: post-cycle contract-consistency classification
  (`sbx/scheduler.py::CONSISTENCY_*`, classify-never-abort).
- **A7 revised:** contracts freeze at the first TSO tick ≥ new
  `MultiTSOConfig.sbx_warmup_s` (default 1800 s) — the pre-loop
  snapshot left standing 20–70 Mvar `q_meas − q_std` offsets on the
  stiff ties and biased the SBX arm at the boundaries; the settled
  snapshot matches within ~1–3 Mvar. Adapter rebases its iteration
  counter at the freeze tick.
- `SBXConfig.voltage_margin_pu` 0.005 → 0.01 (v2.2 item-3 calibration:
  worst observed supporter-side within-cycle shift 6.4 mpu).
- Smoke test extended to three arms (`sbx` / `sbx_inert` = pinning
  without deals / `none`) after the two-arm iteration showed the
  original C3 confounded by the contract pinning itself.

**Result:** `pytest tests/sbx` → 45 passed.  Three-arm 360-min smoke
test **PASSED C1–C8** (500 Mvar sink at bus 15, zone-3 v_min 1.00,
stress minutes 60–210): 4 unilateral paid deals, same-cycle deals on
both zone-3 corridors (v2.2 item 5), settling, full unwind with refs
back at v_std, every TSO solve optimal, supporters exactly clean,
margin ratio 0.64, zero `sign_mismatch` consistency labels.  Findings
F1–F5 (pinning cost +0.354 vs deal benefit +0.019 pu·step; zone-3
joint-box collapse t = 0 under collinear corridor couplings; margin
recalibration; tier-3 attribution warning; border AVRs) in
STATUS_SBX.md §5.4.

**Reason:** Phase 5 closes the loop: the protocol pipeline built in
Phases 1–4 now runs against the pandapower plant through the existing
runner, on the BME-identical 3-area partition, with all coordination
carried by capability messages and frozen voltage references only.
