# 2026-11-04 — Trial weight set of the v3 leader in `run_multi_system_ofo.main()`

**Reason.** Author request: set the current most promising candidate as `TRIAL_PARAMSET`, so
that it can be run as a test in `experiments/run_multi_system_ofo.py`. It is the leader of
procedure v3 (run 0006) before its settling confirmation, not the procedure's output;
SELECTED_PARAMSET is unchanged.

## Changes (≈ 09:00, shared checkout, branch minimal/thesis-core, not committed)

- **New parameter set `configs/paramsets/trial_8ad8_v3_run0006.json`.**
  - Key 8ad82442269b. The ten weights are those of run 0006 (worktree `F:/qOFO_res`,
    `selection/selection_complete_sts+bank.json`).
  - Base of the search: `stage1_v3_uc_2026_11_pq10` (θ_STS,min 0.01 pu).
  - Measured in run 0006 over W1–W4 and W6:
    - e_v,TS 0.00799, e_v,DS 0.00571 (v2 winner: e_v,TS 0.00827);
    - 1 counted tap reversal, 0 bank reversals;
    - banks commit at the W3 outage and the W4 load ramp; couplers move in every window.
  - The procedure's own result will be `ch9_selected_uc_2026_11_v3.json` in the worktree.
- **`experiments/run_multi_system_ofo.py`.**
  - `TRIAL_PARAMSET = "trial_8ad8_v3_run0006"` (was `"ch9_selected_uc_2026_11_v2"`); the
    comment lists the new set.
  - **Fix:** with `SCENARIO` set (it is `'GA'` in this checkout), `main()` now applies
    `TRIAL_PARAMSET` on top of `scenario_config(...)`, in place of the scenario's paramset of
    record.
    - Before, the scenario path ignored `TRIAL_PARAMSET` but printed it as
      "Controller weights". A GA run since 2026-11-03 therefore used the paramset of record
      (`ch9_selected_band_2026_10_a1`: bank price 25.5, g_w_gen 9.4·10⁵), not the trial set
      it named. The run folder's recorded config shows the weights actually used.
    - The Case-2 variants change no weight field, so the overlay is clean. Variant M derives
      `central_dso_g_v` from `dso_g_v`, which the trial set leaves at 10⁴.
  - The `make_config()` path is unchanged.

## Verification

- Both paths built without simulating, checked against the trial file's ten fields:
  0 mismatches.
  - GA / O3: bank price 25.5 → 22.76, g_w_gen 9.4·10⁵ → 1.29·10⁶; start
    2016-01-08 06:00, 360 min, `dso_gamma_oltc_q` 0, `shunt_ff_ledger` True.
- Not simulated here.

## Open

- The settling confirmation of 8ad82442269b runs in the worktree since 08:52. If it fails,
  the procedure takes the next candidate and this trial set is not the selection.
