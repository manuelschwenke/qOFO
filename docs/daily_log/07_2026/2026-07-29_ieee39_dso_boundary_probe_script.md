# Rerunnable IEEE 39 / DSO boundary-condition probe

**Timestamp:** 2026-07-29 16:07:21 +02:00  
**Reason:** Preserve the investigation of unexpectedly low DSO voltages and coupling-transformer OLTC saturation as a reproducible code-based analysis.

## Added analysis package

Created:

- `network/ieee39/analysis/__init__.py`
- `network/ieee39/analysis/probe_dso_boundary_conditions.py`
- `network/ieee39/analysis/README.md`

Default invocation:

```powershell
python -m network.ieee39.analysis.probe_dso_boundary_conditions
```

## Study matrix

The default run evaluates the first 96 native 15-minute profile samples for:

- `coupled`: complete IEEE 39 + four-DSO system, with endogenous 345 kV primary voltages and no distributed slack;
- `isolated`: one independent power-flow case per DSO, with three stiff 345 kV `ext_grid` sources at 1.03 p.u., 0 degrees, and equal `slack_weight = 1/3`;
- DSO DER power factors 1.00, 0.98, and 0.95;
- non-unity DER reactive power using the established inductive convention (`sgen.q_mvar < 0`);
- coupling-transformer `DiscreteTapControl` objects regulating each 110 kV (`mv`) terminal at 1.03 p.u.;
- DC initialization for fresh/retry power flows and result-based chronological warm starts.

## Outputs

The default output directory is:

`results/ieee39_dso_boundary_condition_probe/`

Files:

- `boundary_probe_timeseries.csv`: one row per boundary, power factor, timestamp, and DSO;
- `boundary_probe_summary.csv`: convergence counts, primary/secondary/DSO voltage ranges, OLTC-limit fraction, and interface P-Q extrema;
- `run_metadata.json`: complete boundary, solver, power-factor, and profile configuration;
- `README.md`: human-readable reproduction assumptions.

## Controlled outputs and actuators

- **Controlled output:** 110 kV voltage at each coupling transformer.
- **Actuator:** discrete coupling-transformer tap.
- **Characterization outputs:** aggregate interface P and Q.
- **Boundary variable under investigation:** endogenous coupled-TS primary voltage versus fixed 1.03 p.u. stiff-primary voltage.

## Validation

A two-step smoke test for both boundaries at unity power factor completed successfully and produced the expected contrast:

- coupled case: the first two rural-700 points did not converge;
- isolated case: all eight DSO power flows converged, primary voltages remained exactly 1.03 p.u., and DSO voltages remained within approximately 1.010–1.076 p.u.
