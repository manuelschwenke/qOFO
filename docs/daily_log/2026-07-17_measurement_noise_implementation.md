# Controller-facing measurement noise

**Timestamp:** 2026-07-17 12:12:01 +02:00  
**Reason:** Replace the ideal controller feedback option with selectable, reproducible measurement-uncertainty profiles while retaining the ideal behaviour as the default.

## Implemented profiles

The configured percentages are symmetric error-envelope half-widths, not Gaussian standard deviations.

| Profile | Voltage magnitude | Current magnitude | Active power | Reactive power |
|---|---:|---:|---:|---:|
| `minimum` | +/-0.1% | +/-0.2% | +/-0.1% | +/-1.0% |
| `conservative` | +/-0.5% | +/-0.5% | +/-1.0% | +/-2.0% |

The `minimum` profile uses the best-case endpoints and `conservative` the worst-case endpoints of the previously reviewed accuracy ranges.

## Method and structure

For a true measurement `x`, the controller receives

```text
x_meas = x + xi * r * s,       xi ~ Uniform(-1, 1)
```

where `r` is the selected relative bound. Voltage uses `s = 1 pu`. Current and power use `s = max(|x|, f * rating)` to avoid unrealistically zero error at zero or light loading; the configurable defaults are `f_I = 0.20` and `f_PQ = 0.02`. Nonnegative voltage and current magnitudes are clipped at zero.

- Added `MeasurementNoiseConfig` to `configs/config.py`; noise remains disabled by default for backward compatibility.
- Added `core/measurement_noise.py`, which perturbs only newly constructed controller-facing `Measurement` packets. Pandapower result tables remain plant truth.
- Integrated the model at every measurement hand-off in `experiments/runners/multi_tso_dso.py`, including initialization, centralized control, all TSO zones, all DSOs, capability messages, and tie-line feedback.
- The seeded draw is cached by physical channel within a control instant. Repeated observations of the same physical PCC at that instant share the same realization.
- Perturbed analog channels: bus-voltage magnitudes, branch-current magnitudes, interface reactive power, generator/DER active and reactive power, and tie-line active/reactive power.
- Kept exact: voltage angles, OLTC positions, shunt states, generator voltage setpoints, and DER voltage references. These are digital states/commands rather than the requested analog feedback channels.

Configuration example:

```python
cfg.measurement_noise.enabled = True
cfg.measurement_noise.profile = "minimum"  # or "conservative"
cfg.measurement_noise.seed = 20260717
```

## Evidence basis

- Siemens SICAM P850/P855 specifies operational uncertainty classes including current 0.2, active power 0.5, and reactive power 2 under its stated measurement conditions: <https://cache.industry.siemens.com/dl/files/947/109818947/att_1137821/v1/MAN_SICAM_P850_P855_7KG85xx_US.pdf>
- Siemens SICAM Q200 specifies +/-0.1% of nominal input voltage for voltage magnitude: <https://cache.industry.siemens.com/dl/files/896/109744896/att_1331741/v1/SICAM_Q200_7KG97_MAN_US.pdf>
- Schneider Electric ION9000 lists class 0.1 voltage, current, and active-power measurement performance, illustrating the high-accuracy end of substation-class meters: <https://iportal.se.com/Contents/docs/SQD-METSEION95040_DATASHEET.PDF>
- IEC 61869-1:2023 defines error-limit requirements for analog and digital instrument-transformer signals used for measurement, protection, and control: <https://webstore.iec.ch/en/publication/34049>
- NIST recommends a rectangular/uniform distribution when only practically certain lower and upper limits are known and there is no evidence favoring values within them: <https://physics.nist.gov/cuu/Uncertainty/typeb.html>

These sources establish device/measurement-chain accuracy envelopes. The two repository profiles are deliberately rounded scenario sets for robustness experiments, not a claim that every installed sensor has exactly those errors.

## Verification

- `tests/test_measurement_noise.py`, `tests/test_measurement.py`, and `tests/test_controller.py`: **64 passed, 1 skipped**.
- Python byte-compilation passed for all modified Python files.
- Noise-enabled `run_multi_tso_dso` initialization/pre-loop smoke test passed.

## Assumptions and unresolved risks

- Errors are bounded, zero-mean, independent across physical channels and control instants, with no temporal bias, calibration drift, cross-channel covariance, quantization, delay, packet loss, or bad data.
- Current and power rating floors are modelling choices and should be calibrated to the actual CT/VT, transducer, meter, and SCADA chain when hardware data are available.
- The model injects measurement uncertainty only. It does not alter plant dynamics, load/process uncertainty, cached sensitivities, actuator execution, or communication latency.
- Because reactive-power accuracy depends strongly on power factor and loading, the conservative Q profile should be the primary stress case.
