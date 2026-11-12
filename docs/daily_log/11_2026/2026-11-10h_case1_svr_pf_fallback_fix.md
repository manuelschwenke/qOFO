# 2026-11-10h -- Case 1 SVR failure at the 250 Mvar connection: diagnosis and power-flow fallback fix

Author 2026-11-10 ~18:30: "we need to find out why SVR failed ... and try to fix it, if possible"; "if you can find
out a fix for case 1 svr failure, you can try to rerun it with the fix, unsupervised."

## Failure (rerun base 5c2d8b0, `ch_10_case1_svr/0001_2026-11-10_160316_run_case1_v5rb_A`)

- SVR (code `S1`) failed at step 365, t = 7300 s (121:40). The event: the 250 Mvar connection at bus 11
  (`CONTINGENCY_LOAD|bus11`). The other five Case 1 variants passed it.
- Console sequence:
  1. Post-contingency PF.
  2. Rate limiter held coupler trafo3w#3 (+0 -> -1 undone).
  3. Re-solve with the relays held and no Q limits: converged.
  4. SVR acts.
  5. End-of-step PF with Q limits and all local loops active: diverged.
  6. Old fallback, flat start without Q limits: diverged ("Power flow failed").
  7. The step continued with an unsolved network. The next PF raised LoadflowNotConverged and ended the run; 0
     records saved.
- SVR differs from SVR⁻ (code `S1m`, which survived) only by `svr_rpr_voltage_priority` = True. The safeguard gives a
  different pre-event state; it does not change the solve.
- v5 run of record (old start, ±0.01): SVR survived the event with G3 at its set-point bound 1.05 pu, Q 194 -> 312
  Mvar.

## Diagnosis (`2026-11-10h_diag_svr.py`, run on e5power in `F:\qOFO_runs\diag_svr_case1`, not a run of record)

- SVR was run as `run_case1.run_one` up to the record of step 364 (t = 7280 s), and the plant net was saved
  (`pre_loop_hook` holds the plant's net object; `STEP_HOOK` saved and stopped).
- Pre-event state: G3 296 Mvar of 400, G7 227 of 350, V_set G3 1.05.
- Replay of the connection:

| solve | result |
|---|---|
| A: relays active, Q limits | ControllerNotConverged |
| B: relays held, Q limits (init results) | converges: V_min 0.946 pu (bus 11), G3 at 400 Mvar (V 1.032) |
| C: relays held, no Q limits | converges: G3 would need 443 Mvar |
| Q continuation 25 ... 250 Mvar, relays held, Q limits | converges up to 250 Mvar |

- Established: the plant has a solution after the event, with G3 at its limit. **The failure is numerical.** The
  end-of-step solve with the Q(V) loops and the PV->PQ switching does not converge, and the old fallback from a flat
  start cannot find the stressed solution either. It is not a voltage collapse.

## Fix (branch `res/svr-fix-2026-11-10`, on top of `res/rerun-cen-2026-11-10` = 4a0fa5e)

| File | Change |
|---|---|
| `core/plant.py` | new `runpp_warm_staged(net, res_bus, sgen_q_mvar, **kw)`: from the last converged bus voltages and DER outputs, (1) relays held, Q(V) loops active, no Q limits; (2) Q limits, Q(V) loops active; (3) if (2) fails, Q limits with the Q(V) loops held at their stage-1 output for this solve; returns the stage |
| `experiments/runners/multi_tso_dso.py` | end-of-step PF: snapshot `res_bus` and `sgen.q_mvar` before `plant.advance` (static plant only); on LoadflowNotConverged first `runpp_warm_staged`, the old flat-start unclipped retry only if that fails too; log lines `warm re-solve converged (<stage>)` |
| `tests/test_runpp_warm_staged.py` | 2 tests: Q limits enforced from a snapshot after cleared results; without Q limits the machine exceeds them |

- Only the failure path changes. A solve that converges is untouched and bit-identical.
- Searching all console logs: no completed run of the rerun base or of the v5 runs of record (REF 0004-0006, Case 1
  0001) ever reached the flat-start fallback; only this SVR run did. So the fix changes no completed result.
- Check on the captured state (`2026-11-10h_test_fix_on_capture.py`), event applied:
  - The old fallback (flat start, no Q limits) fails.
  - `runpp_warm_staged` converges at stage 2 (Q limits, Q(V) loops active): V_min 0.9555 pu at bus 11, G3 400 Mvar
    (at limit), G7 251 Mvar, STS IBG Q 109 -> 162 Mvar.
- Tests: the new 2 + relays-held, coupler band, set-point relay, local rules and CEN rule tests, 25 passed; both
  guards pass.

## Rerun

Case 1 SVR alone with this commit, scenario A, as its own run folder (author: "rerun it with the fix,
unsupervised"). Launch and result in the section below.

- Launched 19:35 on e5wimi (one process, pinned to logical CPU 39; the author's limit of 20 workers holds: MC 16 +
  CEN session 1 + this 1). Run folder `ch_10_case1_svr/0002_2026-11-10_193505_run_case1_v5rb_A_svrfix` in
  `F:\qOFO_fix`, tag v5rb_A_svrfix, code 3bc50ba (clean).
- A numbering placeholder `0001_2026-11-10_160316_run_case1_v5rb_A` puts it after the Case 1 run of 16:03.
- Tooling: `F:\qOFO_screens\2026-11-10_rerun_launch_fix` (`launch_svr.ps1`).

## Attempt 1 failed, fix v2 (af5b010)

- 3bc50ba in the rerun (case1/0002_2026-11-10_193505_run_case1_v5rb_A_svrfix): at step 365 the warm re-solve diverged
  too ("invalid value encountered in divide", V / |V| in the Jacobian, i.e. a zero voltage in the warm start). The
  flat-start fallback failed; the run failed as before (kept as the record of the attempt).
- Cause (reasoned, not reproduced offline): the snapshot was taken after `seed_qv_equilibrium`, which writes the DER
  outputs of a linearised closed-loop equilibrium (from the shared Jacobian) and can be far off on the stressed point
  after the connection. The offline check on the capture had no seed, so it passed. A diverged solve that clears
  `res_trafo3w` was a second candidate; the offline test did not reproduce it.
- v2 (af5b010):
  - the snapshot, now `core.plant.snapshot_results` with every `res_*` table and the DER outputs, is taken BEFORE
    the seed;
  - `runpp_warm_staged` restores all of it;
  - unit test with a far-off seed. 15 tests pass.
- Rerun of record `case1/0003_..._run_case1_v5rb_A_svrfix2` (e5wimi, CPU 39, started 20:18) and a diagnostic copy
  (`F:\qOFO_screens\2026-11-10_svr_case1_diag\diag_v2.py`, stops at 7400 s).
- **Result at step 365 (both runs):**
  - "end-of-step PF with enforce_q_lims=True diverged; retrying warm from the last converged state..."
  - "warm re-solve converged (warm, Q limits, Q(V) loops active)". The run continues.

- But the rerun failed one step later: the next step's profile re-solve (`plant.advance`, runner line 4132) raised
  LoadflowNotConverged (50 iterations) in `run_control`'s initial power flow. Same in the diagnostic copy
  (traceback). 0 records saved (case1/0003 kept as the record of the attempt).

## Fix v3 (04c8124)

- Cause: pandapower starts each solve from a DC estimate ("auto" init with `calculate_voltage_angles`). On the
  stressed post-event point that start diverges, although a converged state lies right next to it.
- `PandapowerStaticPlant.advance` now:
  - snapshots the result tables before every solve;
  - if the solve diverges (run_control plants only), re-solves warm with `runpp_warm_staged`;
  - if that fails too, restores the snapshot and raises the original LoadflowNotConverged, so the runner's
    end-of-step fallback (pre-seed snapshot) runs as before.
  - A converging solve is untouched: test against plain `pp.runpp`, max |dV| < 1e-9.
  - A counter `warm_resolves` and a log line `[plant] power flow diverged; warm re-solve converged (...)`.
- Tests: 2 new plant tests (forced divergence rescued; converging solve identical). 33 plant / relay / CEN tests
  pass.
- Rerun of record `case1/0004_..._run_case1_v5rb_A_svrfix3` (e5wimi CPU 39, 20:59) plus a full-horizon
  diagnostic copy (`diag_v3.py`, pickles the net if the run fails). The finisher waits for it.

- Result: failed again at step 366. The plant-level warm re-solve ran and failed too (no "[plant]" line); 0 records.
  The diagnostic copy saved the state the failing solve tried (`v3_failed_net.pkl`, 365 records).

## Fix v4 (af3f5e9)

Offline on that state (`2026-11-10h_analyse_v3.py`), warm start:

| solve | result |
|---|---|
| relays held, Q(V) on, Q limits | converges: V_min 0.9736 pu (bus 11), G3 400 Mvar (at limit), IBG 172 Mvar |
| relays held, Q(V) on, no Q limits | ControllerNotConverged |
| no control, Q limits | converges |
| no control, no Q limits | converges: G3 would need 417 Mvar |

- The re-solve had started with the unconstrained Q(V) solve and given up. The Q-limited solve converges directly,
  but only with 100 Newton iterations; the plant's budget is 50.
- `runpp_warm_staged` now tries, each with at least 200 Newton iterations:
  1. the Q-limited solve, Q(V) loops active;
  2. the route through the unconstrained solve;
  3. the Q limits with the loops held.
- It converges on the saved states of steps 365 and 366 (`2026-11-10h_test_fix_v4_on_capture.py`): G3 at
  397 / 400 Mvar, V_min 0.972 / 0.974 pu. 33 tests pass.
- Rerun of record with af3f5e9 (tag v5rb_A_svrfix4) plus a diagnostic copy; see the next section.

## Open
- The MC campaign running on 5c2d8b0 has the old fallback. An episode that reaches it can still fail there; check
  its failures for "retrying flat-start unclipped" after the campaign.
- This commit goes into the final base (merge after all runs).
