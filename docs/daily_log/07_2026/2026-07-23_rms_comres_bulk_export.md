# RMS replay: native ComRes bulk export and offline sampling

- **Timestamp:** 2026-07-23, after completion of replay run 0047
- **Changed:** `pf/result_export.py`, `pf/plant.py`,
  `experiments/run_rms_phase6_replay.py`,
  `experiments/run_rms_openloop_uy.py`
- **Recovery utility:** `experiments/postprocess_rms_phase6_replay.py`
- **Tests:** `tests/pf/test_result_export.py`

## Reason

Run `0047_2026-07-23_104551` had completed all 45 RMS dispatch intervals
through 900 s and had saved both controller record files.  End-of-run
post-processing was nevertheless taking hours because
`ScreeningContext.read()` performs one PowerFactory `ElmRes.GetValue` call per
sampled result cell and rereads the time column for every signal.  The first
controlled-output pass completed, but the second per-DER pass still required
roughly 1.6 million scalar API calls.  It was interrupted only after the RMS
simulation and records were safely complete.

## Method

1. Export the complete active `ElmRes` once with PowerFactory `ComRes`
   (`iopt_exp=6`, `iopt_honly=0`, `iopt_csel=0`, `iopt_sep=1`).
2. Use the complete export deliberately.  A selected-column ComRes probe on
   PowerFactory 2025 SP4 omitted `b:tnow`; the complete export includes time
   as its first column.
3. Parse the two ComRes header rows and match `(PowerFactory object,
   variable)` pairs to registered monitor labels.  `GetFullName()` paths are
   normalised by removing database-container class suffixes from non-leaf
   path segments; the shorter ComRes object path must then be a suffix of the
   full database path.  This disambiguates repeated leaf names such as
   `QVPRE.ElmDsl`.
4. Read only the mapped numeric columns with pandas, in chunks.  NumPy selects
   every `stride`-th global result row and always appends the final result row
   when it is off-stride.  This keeps memory bounded for longer traces.
5. Keep the original scalar `harvest_trajectories()` for small live-plot
   windows.  Only end-of-run harvesting uses
   `harvest_trajectories_bulk()`, so no bulk export is triggered at every
   dispatch.

## Validation

- The retained run-0047 ComRes file is 195,247,563 bytes with 236 columns and
  approximately 90,000 result rows.
- Native ComRes export time: 78.1 s.
- Production catalogue validation: all 182 requested signals mapped and
  reached exactly 900.0 s:
  - 12 TS/DS interface-Q signals;
  - 82 controlled/PCC/TN voltage signals selected by `u_`;
  - 44 DER park-Q signals;
  - 44 DER park-voltage signals.
- pandas/NumPy loading took about 5 s after the PowerFactory connection.
- Complete recovery (CSV generation, comparison/settling tables, and PNG
  figures) took 51.4 s from the retained ComRes file; no RMS rerun occurred.
- `py_compile` passed for every modified module.
- Focused tests: `3 passed` (ComRes settings, full-path matching,
  decimal-comma parsing, chunk-boundary stride sampling, final-row retention,
  label filtering, and missing-monitor failure).
- The broader event-pool test file currently has one pre-existing stale test:
  it calls `admit_new_events()` without the now-required `horizon_s`.
  The bulk-export tests themselves pass.
- The existing Gate-E replay suite has one separate stale expectation that
  assumes `der_q_capability_override_pu is None`; the runner has intentionally
  used the documented temporary `+/-1.0 pu` ablation since 2026-07-21.
  The remaining replay tests pass.

## Recovered run-0047 result

Run assumptions and constraints:

- profiles enabled on both plants;
- `g_w_dso_oltc = 150`;
- 20 s DSO and 180 s TSO dispatch periods;
- temporary DER capability override `q_min/q_max = +/-1.0 pu`;
- measurement noise and reachability guard disabled;
- the controllers see only cached sensitivities and their own plant
  measurements, never the RMS plant equations.

Actuators were DER reactive-power references, synchronous-machine AVR
references where an AVR exists, 2W/3W OLTCs, and installed shunts.  Evaluated
controlled outputs were the 12 TS/DS interface reactive-power flows and the
three TN-PQ zone-mean voltages.

All evaluated 20 s intervals settled:

- interface Q: 540/540 intervals, maximum settling time 5.072 s,
  95th percentile 0.572 s;
- zone voltage: 135/135 intervals, maximum settling time 5.672 s,
  95th percentile 1.952 s.

Static-versus-RMS independent-closed-loop endpoint errors:

- interface Q: RMSE 3.126 Mvar, MAE 2.187 Mvar, maximum 7.983 Mvar;
- zone voltage: RMSE 0.003073 pu, MAE 0.002491 pu, maximum 0.006098 pu.

These endpoint errors combine plant-model differences with subsequent
controller-trajectory divergence; they are not an open-loop plant residual.
The large dimensionless overshoot fractions in tiny-step intervals should not
be read as large absolute excursions.

## Output

The recovered run now contains:

- `csv/rms_der_raw.csv`;
- controlled-output, endpoint, settling, and actuator-divergence CSVs;
- interface-Q, zone-voltage, actuator, and DSO/TN-voltage figures;
- `postprocess_recovery.json`, explicitly stating that the data came from the
  retained ComRes CSV without an RMS rerun.

## Risks / unresolved points

- ComRes 2025 SP4 emitted six decimal places under the validated settings.
  The resulting quantisation (at most approximately `5e-7` in the exported
  unit) is far below the 1 Mvar interface-Q and 0.001 pu voltage settling
  floors, but a future probe may set an explicit higher export precision if
  raw numerical reproducibility below those scales is required.
- Native CSV generation still scales with result rows times result columns.
  This removes the dominant Python/PowerFactory per-cell overhead, but a
  multi-hour trace can still produce a multi-gigabyte intermediate CSV.
  PowerFactory-side row decimation or a separately exported time vector plus
  selected variables remains a possible later optimisation.
