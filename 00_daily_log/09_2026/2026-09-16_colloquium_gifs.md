# Research-colloquium QSS animations

Timestamp: 2026-09-16T13:32:00+02:00
Reason: prepare two GIF sets showing a four-hour controller run for the research colloquium.

## Assumptions and experimental scope

- User confirmed the frozen thesis network: rural_700, DSO_3 DER and active-load scaling both 2.0.
- Fixed parameter set ch9_selected; its JSON provenance states candidate 399e30d3b60b was reconfirmed on the doubled network on 2026-09-16.
- Existing 2016-01-05 08:00 profile start and disturbance schedule; horizon shortened to 14,400 s. Events after minute 240 remain in configuration but do not fire.
- PI reference uses the established S1 overlay: PI RVR/RPR, voltage-priority safeguard, zero supervisory deadband, local STS Q(V)/coupler OLTCs, held machine-transformer taps, no switched TS shunts.
- Proposed controller uses O3: TS+STS OFO cascade with inherited SBX-H and MSC/MSR integrator dispatch.
- This compares complete architectures, not only the PI versus OFO control laws. The active actuator sets differ.
- Shunt wear remains reported and not bounded, as specified by the selected parameter set.

## Code changes

- experiments/run_colloquium_gifs.py: standalone run harness loading the existing selected configuration, applying S1/O3 overlays, asserting every selected weight is unchanged, and saving immutable case-specific configs, logs, network metadata, status and records.
- visualisation/colloquium_gifs.py: seven GIF exports with progressive time reveal, fixed axes, event markers and clock; corresponding final PNGs and a ZIP.
- tests/test_colloquium_gifs.py: checks measurement gaps, channel identity, group mapping, causal frame reveal and GIF round-trip metadata.

Method: render the measurement channels already used by TSOControllerLivePlotter and CascadeDSOLivePlotter; no controller model or plant architecture change. Metered values are pre-control snapshots; commands and discrete positions use recorded states. Missing measurements remain NaN. Pilot measurement is reconstructed from the zero-deadband diagnostic V_ref - e_pilot.

Outputs: results/research_colloquium_gifs/0001_2026-09-16_132331

## Validation and outcome

Simulations and visual QA in progress. Append final results here after both runs and all seven GIFs are verified.

## Author correction and data-channel audit

Timestamp: 2026-09-16T13:54:17.166786+02:00

The author recalled the agreed 3 h shunt dwell and requested proof. The original saved config used 1800 s, passed directly to ShuntBankConfig at runner line 3237. Recorded zone 2 bank 5 switches at 120 and 150 min independently confirm the 30 min dwell. Original OFO is retained for audit. New run: results/research_colloquium_gifs/0002_2026-09-16_134936, explicit shunt_int_t_dwell_s=10800. The PI trajectory has no shunts and is reused with original raw provenance.

Found an existing recording difference: S1 does not populate zone_v_meas / zone_q_*_meas, while OFO TS snapshots occur only on 81 dispatch ticks. Final plots therefore explicitly use post-power-flow voltage and Q telemetry for both cases, retaining liveplot layout/aggregation. Pilot controller reading comes from V_ref - e_pilot. This changes the plotting convention, not any controller or plant implementation.

Added analysis/colloquium_run_report.py: timeline, status, physical voltage ranges, movement counts, and a per-bank audit enforcing 10800 s between observed OFO shunt commits. Seven renderer checks pass after the change.

## Corrected OFO outcome and initial visual QA

Timestamp: 2026-09-16T14:14:32.448905+02:00

Corrected OFO completed 720 samples through 14400 s; TS: 243 optimal solves; STS: 2880 optimal solves. Configuration comparison shows only result_dir and shunt_int_t_dwell_s changed. Observed repeat switches: zone3/bank1 at 18 and 198 min, zone2/bank3 at 30 and 210 min. All recorded intervals meet 10800 s. Seven total bank-position changes.

Four independent OFO GIFs generated, decoded and visually checked. Fixed integer y-axis ticks for constant positions, numeric transformer ordering and stable animation palette. Nine targeted renderer checks pass. Encoded interface GIF inspected at 0, 30, 120 and 240 min: no premature data/event reveal. PI is still running; final three GIFs and ZIP are queued in the run-local finish_exports.py process.

## 2026-09-16 — ofo/01_ts_voltages rendered without PI records

- **Changed:** `visualisation/colloquium_gifs.py`, `main()`: when `--only` lists nothing but `ofo/` figures, only OFO records are loaded now. Before, `ofo/01_ts_voltages` was left out of this rule.
- **Effect:** When only OFO records are loaded, the TS voltage y-axis comes from OFO data alone. When a `pi/` figure is selected too, both cases are still loaded and share one scale, as before.
- **Reason:** The PI reference run (`0001_.../pi`) was still simulating, so `finish_exports.py` was waiting and the OFO TS voltage GIF was missing for the colloquium.
- **Output:** `results/research_colloquium_gifs/0002_2026-09-16_134936/ofo/01_ts_voltages.{gif,png,gif.json}` (361 frames, 1600x900, 38.1 s). This run overwrote `gif_validation_partial.json`, which now lists only this GIF.
- **Pending:** When PI finishes, `finish_exports.py` re-renders `ofo/01_ts_voltages` with the shared PI/OFO scale and overwrites this version.

## 2026-09-16 — OFO variant with the generator outage on G5 (run 0003)

- **`experiments/run_colloquium_gifs.py`:** `--prepare` has two new options. `--trip-gen <idx>` moves the one scheduled generator trip/restore pair (30/180 min) to another `net.gen` index; it fails unless exactly one such pair exists. `--cases pi,ofo` chooses which cases are prepared. `design.json` now records `cases` and `generator_outage_index`. Default behaviour is unchanged.
- **`visualisation/colloquium_gifs.py`:** new `events_from_config(cfg)` builds the event marker labels from `cfg.contingencies` (up to 240 min). `main()` overwrites the global `EVENTS` with it and fails if the loaded cases have different schedules. Run 0002 still gets the same labels. 9/9 tests in `tests/test_colloquium_gifs.py` pass.
- **Reason:** the author asked for the OFO case with an outage of G5 (TS area 3, `G7_bus35`, 464 MW) instead of G2, and all GIFs for it.
- **Run:** `results/research_colloquium_gifs/0003_2026-09-16_144901/ofo`. Comparing configs field by field against 0002/ofo, only the two gen-event indices and `result_dir` differ. 720 records; TSO 243/243 and DSO 2880/2880 optimal. The audit passes (`run_validation.json`): no TS voltage-bound violations, 3 h shunt dwell respected. The runner also takes machine transformer 11 out of service with G5.
- **GIFs:** all five OFO GIFs rendered and verified (361 frames, 1600x900).
- **Observation:** the largest difference from 0002 in TS area voltages is 0.0073 p.u. (area 3 max, t = 33 min). G5's reactive infeed during the outage is taken over by G2 and DER in area 3.
