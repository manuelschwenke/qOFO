# Band campaign: 301c285ee52b settling run aborted; STS margin check queued

Timestamp: 2026-10-27, ~09:00 Europe/Berlin.

Campaign: `results/THESIS_ch8_weight_selection_band_20261025/` (procedure
`docs/tuning/2026-10-25_weight_selection_procedure.md`). No repository code
changed; campaign-folder scripts only.

## 1. Settling run of 301c285ee52b aborted (author request)

- **State:** the settling check ran ranks 1–4 in parallel (4 × 7 workers):
  fde2e81af6b1, c08b1d3d33ab, e27ce5e4f01e, 301c285ee52b. At 08:51,
  301c285ee52b had 47 / 120 steps done, and 6 of them had a logged
  `n_inner` > 27.
- **Precedent:** in the 2026-10-24 application, 301c285ee52b failed with a
  p95 of 31.0. The logged `n_inner` predicted all four verdicts there:
  p95 n_inner 19 vs 30–31, against criterion p95 22.05 vs 31–33.
- **Action:** killed its process tree (`taskkill /T /F`). The driver logged
  `exit 1`, and `criterion()` will record it as FAIL with `error: no steps.csv`.
  This is **not a measured failure**; `expC/ABORTED_cand_301c285ee52b.txt`
  states so.
- **Effect on the procedure:**
  - None if any of ranks 1–3 passes, since the checks run in rank order.
  - If all three fail, the driver would skip 301c285ee52b and move to
    rank 5+. 301c285ee52b would then have to be re-run by hand before the
    result is accepted.
- **Effect on the ETA:** none. The driver waits for the slowest run
  (c08b1d3d33ab, e27ce5e4f01e). The kill frees 7 cores.

## 2. STS voltage-margin check queued (author request)

- **Reason:** the stored Stage-II rows keep no voltage extremes. The leading
  candidate fde2e81af6b1 has e_v(DS) +74 % vs the Stage-I point. Without the
  band it violated the 0.90–1.10 pu corridor in W1.
- **New files:**
  - `margin_unit.py` re-simulates one (key, window) unit with full records.
    It uses the same construction as `ch_9_3_stage2_screen._run_one_window`:
    the Stage-I paramset with the weights overlaid, frozen network, evented
    bank, 12 h, `make_config` including the band, and BLAS pinned to 1 thread.
  - `margin_report.py` reports per STS: v_min and v_max; the margin to
    0.90/1.10; the largest exceedance of the band and the minutes outside it;
    e_v(DS); demand; tracking RMS; DER utilisation; coupler taps per day.
    It also gives the TS e_v and margin, and checks each run's e_v(TS)
    against the stored row.
  - `followup_margin.py` (pid 42944) waits for `FOLLOWUP_DONE.txt`, the
    Experiment C reference. It then runs 10 units: the result key from
    `DONE.txt` and the Stage-I point 3cb66225e1a7, in W1–W4 and W6, ≤ 30
    workers. Finally it writes `margin_check/margin_report.txt` and `.json`,
    then `MARGIN_DONE.txt`.
- **Tests:**
  - `margin_report.py` on the 2026-10-24 records (c08b1d3d33ab vs Stage-I,
    no band) reproduces the earlier finding: DSO_2 1.0958 pu in W1, 84 min
    above 1.08. 10 / 10 runs reproduce the stored e_v(TS).
  - A 0.2-h smoke run of `margin_unit.py` (fde2e81af6b1, W1) confirms the
    band (0.92, 1.08) is in the config.
- **Code state:** the working tree equals the campaign's launch state.
  HEAD d9633fe is docs-only on top of 9d03c34, and the uncommitted diff is
  identical to `code_state/tracked_changes.patch`. The re-simulation should
  therefore reproduce the stored rows exactly; the report checks this.
