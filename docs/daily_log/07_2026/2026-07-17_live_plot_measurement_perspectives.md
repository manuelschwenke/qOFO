# Live-plot measurement perspectives

**Timestamp:** 2026-07-17 (Europe/Berlin)

## Reason

Make the information boundary in the online feedback optimisation study explicit:
controller-facing plots must not expose fictional exact plant states when the
controllers receive noisy measurements.

## Implemented structure

- Added separate pre-control noisy TSO and DSO observables to
  `MultiTSOIterationRecord`; existing post-control plant fields remain unchanged.
- Captured the same noisy `Measurement` objects passed into each OFO controller
  and derived plot-ready voltage, current-loading, P/Q, tie/interface-Q, and
  perceived reactive-reserve quantities from those packets.
- Changed the Multi-TSO controller plot to show only noisy analogue metering.
  AVR setpoints, OLTC positions, and shunt states remain exact because they are
  issued commands or digital read-back states.
- Changed the cascade-DSO plot to show noisy voltages, currents, DER Q, and
  interface Q. TSO Q setpoints, DSO capability reports, and OLTC positions remain
  controller communication/digital-state quantities.
- Kept the system power-flow plot unchanged: it remains exact post-control plant
  truth.
- Extended tracking/reserve plots with both perspectives: exact post-control
  trajectories are solid lines and noisy pre-control controller samples are point
  markers. The title and section bands state the timing explicitly.
- Kept the SBX plot data path unchanged. SBX continues to consume the noisy zonal
  TSO measurements used by `SBXRunnerAdapter.on_tso_step`.
- Corrected exact TS voltage RMS tracking to use each controller's current
  per-bus voltage reference (including dynamic schedule changes) instead of the
  static scalar fallback.

## Assumptions and constraints

- Analogue measurement noise applies to voltage magnitude, line current,
  generator/DER P and Q, interface Q, and tie-line P/Q according to the configured
  measurement-noise profile.
- Commanded AVR references and discrete OLTC/shunt states are not analogue sensor
  channels and are therefore not perturbed.
- Controller and cascade samples are pre-control; system truth is post-control.
  They must not be interpreted as simultaneous operating points.
- Local-DSO comparison modes do not construct an OFO `Measurement` packet. Their
  cascade analogue traces therefore remain empty rather than falling back to
  plant truth.

## Validation

- Python syntax compilation passed for the record, runner, helper, and three
  modified plotter modules.
- `pytest -q tests/test_controller.py tests/test_live_plot_measurement_perspective.py
  tests/test_measurement_noise.py`: 64 tests passed, 1 skipped.
- A field-read audit confirmed that the TSO/cascade plotters no longer read the
  corresponding exact analogue plant fields; system and SBX plotters were not
  rewired to the new metering fields.

## Open point

If noisy analogue traces are desired for local-controller baselines, define a
separate local-controller sensor model first. Creating synthetic noisy plot data
that the local controller did not actually consume would blur the information
boundary established here.
