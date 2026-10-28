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

## Evening (≈16:00–17:30): merge, WMI, screening, S2

- **Merge of `minimal/thesis-core`** (tuning session: folder renamed to `ch_09_parameter_selection`,
  band set selected). One import path in `run_case1.py` followed the rename. The selected set is
  now `ch9_selected_band_2026_10`, with **λ_max_B = 1.467**, the value the author recalled. M's
  loop-gain target follows it automatically. The 1.414 quoted earlier belonged to
  `ch9_selected_final_2026_10`.
- **WMI on e5power** (author granted Execute Methods, Enable Account and Remote Enable on
  `ROOT\CIMV2`): WMI reads work, and a process started with `Win32_Process.Create` from a WinRM
  session survives the session. e5power has **18 physical cores** (36 logical), not 30.
- **Screening of the reference candidates** (L2 and O2, 360 min each, all `ok`; O2 still on the
  previous paramset, L2 weight-independent):
  - A: L2 area 2 outside [0.95, 1.05] 67 % of the time; each area is hit by one event.
  - B: mild (4.5 %); the line trip barely registers.
  - C: hardest in area 2 (71 %); area 3 is hit only indirectly.
  - Proposal to the author: A.
  - W (withdrawn interface support) is non-zero without STS controllers (70–89 Mvar·min
    after a machine trip, from tap relays and profile drift), so it is read only as a paired
    difference (O3 − O2, S2 − S1).
- **S2 implemented** per the approved design note:
  - Q_0,j in the CAIR message;
  - SVR coupler alignment and the one-time s_p,a coupler term;
  - runner capability inbox and set-point dispatch at the RVR updates;
  - interface flows in the noisy view;
  - guard pairs (S1,S2) and (S2,O3).

  Tests: `tests/test_svr_sts_access.py` (6), 164 regression tests pass. 10-min smoke on
  e5power `ok`:
  - s_p,a zone 2 0.0836 → 0.140, zone 3 0.102 → 0.128;
  - all 12 couplers receive references, 100 % of them inside CAIR;
  - final tracking error 0.03–0.92 Mvar.

## Reference run of record (18:18–21:59, e5power)

Run `results/ch_11_case2_multisystem/ref/0006_2026-10-27_181840_run_ref`:
- code 20873f4 on `case2/ref` (clean), paramset `ch9_selected_band_2026_10`;
- scenario A; launched detached via WMI;
- all six variants `ok`, 1080 records each;
- figures and CSV data in `figures/` (`figures_ref.py`).

The numbers below are one trajectory with no statistics. Units: mpu, RMS over the areas.

| | L2 | S1 | S2 | O2 | O3 | M |
|---|---|---|---|---|---|---|
| e_v | 22.0 | 12.7 | 11.4 | 8.1 | **6.8** | 7.8 |
| e_v,p | 17.9 | 4.3 | **3.0** | 4.6 | 3.5 | 3.7 |

**Access (same law).**
- S1 → S2: e_v −10 %, e_vp −30 %.
- O2 → O3: e_v −15 %, e_vp −24 %.
- Area 2 under O3: R+ min 483 Mvar (O2: 377), u_q 0.071 (O2: 0.198).
- Recruited flow: O3−O2 11–43 Mvar per coupler; S2−S1 10–29 Mvar.

**Price of access.**
- f_V,STS: O3 11–27 mpu vs O2 5–8 mpu.
- O3's DSO_4 leaves its 0.92–1.08 operating band by 4.9 mpu, but stays within the hard limits.
- O3 does not reach its references in DSO_2 and DSO_4, which sit at the band (5–15 Mvar short).

**Law, with the access held (S2 → O3).**
- e_v −40 %.
- The pilot error is higher in area 1 (4.6 vs 3.3 mpu), the area-1 pilot offset of Case 1 again.

**Inter-dispatch interval (Δe_post, W relative to the no-access pair).**
- O3 after the 250 Mvar step: area 2 +2.8 mpu, area 3 +0.9 mpu in 80 s. W = 55 Mvar·min against 5 for O2.
- O3 after the generator trip and after the line trip: ≤ 0.2 mpu.
- The classical schemes after the generator trip: area 3 +12.0 mpu (S1) and +9.5 mpu (S2), with the safeguard on. Once the event lies off the dispatch grid, their withdrawal reappears.
- To be discussed with the author before any remedy (thesis todo).

**M.**
- With the band weights λ_M = 3.55, κ = 2.42 → 1.467.
- **Not a bound:** e_v 7.8 > O3 6.8.
- It uses the STS more than O3 (u_q,STS 0.54–0.73 vs 0.42–0.65). The aggregation prediction "M reaches for transmission resources first" is not supported by this run.
- **No bank switched**, so its commit path is still unexercised.

**Sawtooth in O2/S1/S2.** The period is T_TS during load ramps (60–90, 270–300 min).
- Attribution: the TS actuators move only at dispatches, in same-sign steps with 0 reversals; between dispatches only the machines follow the drift. This is the ramp lag of a 180 s supervisory loop, **not** droop re-anchoring and not hunting.
- O3 has none: its STS-OFOs move 286 Mvar between dispatches against 51 at them.

## M investigated (author request, 22:00–23:30)

**Why M switched no bank** (diagnostic over 360 min,
`0007_*_m_tuning/m_bank_diagnostic.json`):
- the banks were free in every step (holding time, limits and exclusion never bound);
- the bank gradient reached 2.87 per Mvar at the load step and 1.73 at the generator
  trip, the range in which the cascade switches (1–3);
- M's bank price is the selected bank weight × κ (2.42), and the gradient falls within
  20–60 s as M's continuous actuators act.

**Why M is not better than O3: the loop gain at the selected weights.**

| counting rule | λ_max(M) |
|---|---|
| every column | 3.55 |
| Ch 9 rule (MSC/MSR exclusion, as `tuning.stage1_screen.lambda_max_B`) | 3.00 |
| continuous actuators alone | **0.98** |

- The stiff mode is 39 % machine-transformer taps and 33 % banks.
- `int_cooldown` = 5 steps, i.e. 15 min for the 180 s TS-OFO but 100 s for M, so M moved the
  transmission devices nine times as often as the cascade.

**Candidates** (code 0bc6c52; same scenario A, O3 re-run on the same code and identical to
the run of record):

| | e_v | e_vp | worst A2 | STS min margin | TS taps / banks |
|---|---|---|---|---|---|
| O3 | 6.83 | 3.46 | 75.1 | −4.9 | 3 / 5 |
| M (κ 2.42, all columns) | 7.79 | 3.66 | 57.3 | −0.5 | 2 / 0 |
| M_ch9 (κ 2.04) | 7.69 | 3.57 | 56.3 | −3.8 | 3 / 2 |
| M_disc (taps/banks × 4.11) | 7.39 | 3.14 | 57.7 | −0.0 | 1 / 0 |
| **M_rate** (κ 1, TS taps/banks cooldown 45 steps = 15 min, λ over the continuous part 0.98) | **6.25** | **2.98** | **45.7** | **+3.4** | 11 / 8 |

- M_rate is better than O3 in every area.
- Its eight bank commits (all MSCs at 30.7 and 121.7 min) exercise the central commit path
  without failure.
- It moves more: AVR travel area 1 0.16 vs 0.04 pu.
- It also uses the STS slightly more than O3 (u_q,STS 0.48–0.70 vs 0.42–0.65).
- Proposal to the author: M_rate as the definition of M ("same weights, the cascade's
  timescale for the transmission devices, more information"). Not adopted yet.

## 2026-10-28 ~09:30: M adopted (author decision)

**Definition.** M_rate becomes M, with the cooldowns made exactly equal to the cascade's:
- **Transmission taps and banks** are locked after a switch until iteration k + 1 + n with
  n = (1 + int_cooldown)·T_TS/T_STS − 1 = 53, so they are free again after 18 min as in the
  TS-OFO (`variants.ts_equivalent_cooldown`, derived from the config).
- M_rate had 45 steps, i.e. 15.3 min.
- The banks' 30-min holding time applies on top in both M and the cascade.
- The coupler taps were already equal (2 min); the STS-OFO and M both step every 20 s.

**Rest of the definition.** Selected weights, κ = 1, loop gain checked over the continuous
actuators (0.98). The candidates are retired; their record is run 0007.

**Check run** `0008_*_m_check` (06d403b):
- `ok`, identical to M_rate: e_v 6.25 and e_vp 2.98 mpu, 8 bank commits.
- In scenario A every tap or bank move is ≥ 30 min after the previous one, so the lock never
  binds there.
- The lock itself is covered by a unit test (`tests/test_central_updater_wrapper.py`): TS
  devices locked 53 steps, coupler taps the global lock.

**New reference run of record** with all six variants on 06d403b, launched 09:37. It
supersedes 0006 (old M).

## Not done / open

- The reference scenario: the author's choice (A proposed).
- M: re-check κ and λ_max with the band set, and a run with a bank switch, before M leaves
  `NOT_READY`.
- Worker-scaling test on e5power (18 cores), then the reference run of record.
