# Case 2 (`ch:case3`): reference-run infrastructure, noise for every controller, records

Timestamp: 2026-10-27, ~11:00–14:00 Europe/Berlin. Branch `case2/ref` in the worktree
`.claude/worktrees/case2-ref`. The main tree is untouched: the band campaign's follow-ups
import from it.

Reason: the approved plan for the Case-2 simulations
(`~/.claude/plans/i-now-want-to-composed-island.md`), phases 1.1–1.9. None of the six
variants of `tab:case3:variants` ran through a Case-2 driver.

Author decisions of 2026-10-27 that this implements:
- minimum-class metering for every controller, the SVR and the droop included;
- events between two TS dispatches;
- `SELECTED_PARAMSET` from the Chapter-8 tuning session, never restated.

## What changed

| File | Change | Why |
|---|---|---|
| `experiments/ch_10_case1_svr/run_case1.py` | `PARAMSET = SELECTED_PARAMSET` (imported; was the fixed name `ch9_selected`, which lacks `g_w_tso_shunt`) | unblocks L2/S1/O2 and `run_mc`; the run records name and sha256 |
| `experiments/ch_11_case2_multisystem/variants.py` (new) | L2, S1, O2, O3, M on one base; **no event inherited from `make_config`**; noise seed per run; `STEP_ALLOWED` + `assert_case2_controlled`; `NOT_READY = {M}` | one controlled registry; `make_config`'s events drift with other campaigns |
| `configs/config.py` | `MeasurementNoiseConfig.apply_to_svr`, `.apply_to_der_voltage` (default **off**: earlier runs unchanged; Case 2 sets both on) | the SVR and the droop read plant truth |
| `core/measurement_noise.py` | `MeasurementNoiseModel.result_view` (noisy `res_bus`/`res_gen`/`res_sgen`, same channel keys and sample id as the OFO packets); `NoisyResultView`; `DerVoltageMeter` (per-converter VT/meter channel, sample id = plant step); `measured_der_voltage` (single read point) | equal information for every variant |
| `controller/der_qv_local_loop.py`, `core/plant.py`, `experiments/helpers/plant_io.py` (3 sites) | droop V and the dispatch anchor read through `measured_der_voltage`. Anchor and running value share a channel, so a persistent bias cancels in V − V_anchor | one channel for law and anchor |
| `experiments/runners/multi_tso_dso.py` | meter installed on the net after build, sample set per step; the SVR gets the noisy view; **`run_multi_tso_dso` is a wrapper that attaches `partial_log` to any escaping exception** (impl: `_run_multi_tso_dso_impl`); `_zone_voltage_error_inputs` extracted (one definition for the end-of-step error and the new pre-control sample) | metrics need these inputs; failed runs lost all records |
| `experiments/helpers/records.py` | new fields: `gen_q_min/max_mvar`, `zone_gen_idx/in_service/is_slack`; per-unit STS IBG `dso_der_*` with the plant's own capability law; `pre_control_bus_vm_pu`, `pre_control_zone_v_rms_err_pu`, `pre_control_trafo_q_mvar`, `events_fired` at event steps | u_q, R±, u_q,STS, Δe_post baseline, withdrawn support |
| `experiments/ch_11_case2_multisystem/metrics_case2.py` (new) | area metrics (Case 1's + u_q, R±, banks), aggregate = RMS over areas, STS metrics, interface error (full RMS + end-of-interval), recruited flow, event metrics (Δe_post from the pre-control baseline to the step before the next dispatch; withdrawn interface support W) | §1.4 of the plan |
| `experiments/ch_11_case2_multisystem/ref/scenarios.py`, `ref/run_ref.py` (new) | candidate schedules A/B/C (one machine, one line, one load step; offsets 40/100/140 s into the interval) + `thesis6`; producer via `thesis_runs.start_thesis_run(target="ch:case3:nominal")`, one process per variant, full `log.pkl` (partial on failure), `metrics.json`, `events.csv`, `report.md` | the reference run, documented per the results convention |
| `docs/architecture/2026-10-27_S2_design_note.md` (new) | S2 design for the author's review | S2 is an architectural addition |
| tests | `tests/experiments/test_case2_variants.py` (7), `test_case2_metrics.py` (8), `tests/test_measurement_noise_coverage.py` (8) | |

## Verification

- 84 tests pass: plant static parity, droop dead band and local loop, measurement noise (old
  and new), reachability, DSO band, `thesis_runs`, Case 2.
- 10-min smoke runs, events between dispatches:
  - L2 and S1 `ok`, 30 records each; S1 with the noisy view;
  - O3 stopped on a cp1252 console print in the ad-hoc smoke script, not in the code;
    `run_ref` sets `PYTHONUTF8` for its workers.

## e5power (14:00)

- **Env.** `F:\python_environments\qOFO_clean` was copied from e5wimi as a zip over WinRM, to
  the same prefix (author's request). Python 3.12.14 with the reference pins (numpy 2.3.5,
  scipy 1.16.3, pandas 2.3.3, pandapower 3.4.0, cvxpy 1.8.1, gurobipy 12.0.1, networkx 3.6.1,
  numba 0.65.1). Gurobi accepts the academic licence of mschwenke (ID 2830712, expires
  2027-06-03).
- **Code.** `git bundle` of `case2/ref` → clone `F:\qOFO_runs\qOFO_case2` at beeaf87. A rename
  to `F:\qOFO_runs\qOFO` was refused (a handle in the folder); the next push clones into
  `qOFO` directly, so `thesis_runs` records the project as `qOFO` again.
- **Smoke, 10 min of scenario A, single worker per WinRM session:** L2, O2, O3, S1 `ok`,
  30 records each. Wall time incl. init on e5power: L2 1.8 min, O2 1.8, O3 1.6, S1 6.2.
  Run folders `results/ch_11_case2_multisystem/ref/0002…0005_*_smoke_e5power_*` on e5power,
  with README, manifest, header and report; provenance shows beeaf87, clean, paramset sha256.
- **Four workers inside ONE WinRM session were killed** within seconds (BrokenProcessPool,
  empty console logs), consistent with the per-shell memory/process quota. One worker per
  session works.
- **Detached processes cannot be started from WinRM** (breakaway refused), and WMI is denied
  for mschwenke. Long campaigns therefore need one of: WMI rights, a scheduled-task runner
  with a stored password, or local admin (author's choice).

## M rebuilt (14:30–15:30, author decisions)

- `controller/central_controller.py`: the parent's bank-column V² rescaling is kept and
  wrapped (`_AppendFrozenColumns`), so the frozen 3W columns are re-appended. It had been
  disabled on the stale assumption "no shunts", and with the banks in the MIQP the old M
  could not solve at all (`set_shunt_bank_state()` was never called).
- Runner, central step: bank bounds before the solve, then the rank-1 update and the commit
  after it, as in the zone path. M's commits go to `rec.shunt_commits` (zone "central").
- `lambda_max(M)` is computed at initialisation and refused if ≥ 2. The first record carries
  `controller_lambda_max` of every acting controller (and `central_gw_kappa`).
- At the selected weights **λ_max(M) = 5.11**: the single loop does not contract. Two earlier
  attempts to fix this did not work:
  - matching the zone TS-OFOs of the same run reads 2.61, because in M's configuration those
    controllers have full-network sensitivities and extra rows;
  - so the target is now `central_lambda_target` = the paramset provenance `lambda_max_B`
    (1.4136).
- Result: κ = 3.615, λ_max 5.111 → 1.414. 10-min smoke `ok` on e5power, 8 banks in the MIQP.
  M stays in `NOT_READY` until a run with a bank switch confirms the commit path.
- M weights the STS buses with the paramset's `dso_g_v` (2186) instead of the default 2e4.

## Not done / open

- S2: the design note waits for the author.
- M: a run with a bank switch is outstanding before M leaves `NOT_READY`.
- Reference scenario: screening of A/B/C with L2 and O2 running on e5power; the author picks.
- e5power: the author grants mschwenke WMI rights (Remote Enable + Execute Methods on
  `root\cimv2`). Long runs are then launched detached via `Win32_Process.Create`.
  Until then: one worker per WinRM session, held open from e5wimi.
