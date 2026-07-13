# 2026-07-07 — SBX Minimal Phase 0: reconnaissance

**Task:** SBX Minimal build plan v2, Phase 0 (reconnaissance, gate).
**Changed:** New files only — `STATUS_SBX.md` (repo root; findings, corridor
table, ASSUMPTIONS A1–A10, gate questions G1–G5) and this log entry. No code
touched.

**Method:** Static inspection of `core/`, `controller/`, `optimisation/`,
`network/`, `experiments/`; one recon script (scratchpad, not committed)
building `build_ieee39_net(scenario="wind_replace")` + fixed 3-area partition
+ `BoundaryTopology` + converged `pp.runpp` to extract the corridor table.

**Key findings:**

1. `rep1()` does not exist as code — convention only (`docs/BME_SPEC.md` §8).
   To be created as `sbx/fail.py::rep1()` (gate G1).
2. No formal `Plant` protocol; plant interface = `pp.runpp` +
   `measure_zone_tso` + `apply_zone_tso_controls` (PF mirror in
   `core/pf_adapter.py`).
3. Tie-line Q is measured at the in-zone endpoint in load convention
   (`core/measurement.py:429-458`); tie-line **P is not measured** — SBX
   Step 1 needs it → additive optional `Measurement.tie_line_p_mw` proposed
   (gate G2, hard-rule-5 stop-and-report).
4. Tracked outputs: `v_setpoints_pu` + scalar `g_v` per zone;
   `receive_tie_coordination` / `update_voltage_setpoints` are the runtime
   update precedents → `w_track ≡ g_v` (gate G5).
5. Capability LP maps 1:1 onto `MIQPSolver._solve_qp` with `G_w = 0`,
   `G_z = 0` (hard-constraint branch), `alpha = 1`; wrapper unmodified;
   caller must assert `is_optimal`.
6. Timing: `dt_s = 20 s`, `tso_period_s = 180 s` (005 config, reused by
   011/012) → `k_sched = 5` TSO iterations per 15-min cycle (gate G4).
7. Corridor topology of the fixed 3-area IEEE 39 partition is a **triangle**
   ((1,2), (1,3), (2,3); 5 tie lines: 2, 14, 25, 5, 18) → 3-area excluded
   from v2 scope per plan; **no 2-area partition exists** → merge required;
   recommendation A = zone 1, B = zones 2∪3 (gate G3).
8. `net.f_hz = 60` for the IEEE 39 build — the tie-line model must read the
   frequency from the net (a 50 Hz recon assumption was corrected).
9. MC harness: 012 reuses the 006 scenario generator (paired scenarios,
   drop-and-replace); SBX Phase 7 mirrors this with arms
   {AUTONOMOUS, SBX, BME}.

**Reason:** Plan v2 §4 Phase 0 mandates locating the integration points and
confirming corridor topology before any SBX code is written; the gate exists
because the 2-area configuration choice and two interface questions (rep1
helper, tie-line P measurement) need Manuel's sign-off.

**Status:** Paused at the Phase 0 gate. Phase 1 (tie-line + corridor model,
golden tests) starts after G1–G5 are answered.
