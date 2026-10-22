# Full-year rural-700 DSO P-Q characterization

**Timestamp:** 2026-07-29 16:18:51 +02:00  
**Reason:** Generate the full-year CSV and diagnostic scatter plot for the 700 MW-per-DSO scenario after validating the isolated stiff-primary boundary condition.

## Assumptions

- Four synthetic DSOs are solved independently.
- Each DSO has three stiff 345 kV `ext_grid` sources:
  - voltage magnitude: 1.03 p.u.;
  - voltage angle: 0 degrees;
  - distributed-slack weight: 1/3 per source.
- The transmission system is excluded from the characterization power flow.
- All three 345/110/20 kV coupling transformers per DSO retain their physical tap range and 1.25% step.
- Pandapower `DiscreteTapControl` regulates each 110 kV (`mv`) terminal at 1.03 p.u.
- Installed DER: 700 MW per DSO.
- DER active power follows the native PV3, WP7, and WP10 profiles.
- Selected DER reactive-power policy: unity power factor, `Q_DER = 0`.
- Load active and reactive power follow their native 15-minute profiles.
- Fresh and retry power flows use DC initialization; chronological warm starts use previous results.
- `distributed_slack=True`; voltage-dependent loads remain enabled.

## Results

- Profile horizon: 2016-01-01 00:00 to 2016-12-31 23:45.
- Native samples: 35,136.
- DSO power flows: 140,544.
- Converged: 140,544.
- Failed: 0.
- Retries: 0.
- Runtime: approximately 19.8 minutes.
- Time-series CSV rows including header: 140,545.
- Failure CSV rows including header: 1.

| DSO | P min [MW] | P max [MW] | Q min [Mvar] | Q max [Mvar] | V min [p.u.] | V max [p.u.] |
|---|---:|---:|---:|---:|---:|---:|
| DSO 1 | -389.070 | 242.893 | 68.494 | 140.534 | 1.001 | 1.051 |
| DSO 2 | -379.954 | 245.118 | 63.340 | 157.696 | 0.986 | 1.063 |
| DSO 3 | -393.822 | 241.795 | 71.158 | 131.624 | 1.009 | 1.044 |
| DSO 4 | -363.246 | 249.418 | 54.038 | 190.138 | 0.959 | 1.079 |

No sample had a DSO voltage outside 0.9–1.1 p.u.

The interface convention is positive P/Q for import from the stiff primary
sources into the DSO. Negative P therefore denotes DSO export.

## Generated artifacts

Directory:

`results/annual_dso_pq_characterization_isolated_rural_700/`

Important files:

- `annual_pq_timeseries.csv`: auditable complete table;
- `annual_pq_scatter.csv`: combined P-Q table;
- `pq_scatter_DSO_1.csv` through `pq_scatter_DSO_4.csv`: two-column TikZ/PGFPlots inputs;
- `annual_pq_summary.csv`: statistics quoted above;
- `annual_pq_characterization.png`: diagnostic plot;
- `annual_pf_failures.csv`: header only because no power flow failed;
- `run_metadata.json` and `README.md`: reproducibility record.

## Controlled outputs and actuators

- **Controlled output:** 110 kV coupling-terminal voltage.
- **Actuator:** local discrete OLTC tap.
- **Uncontrolled characterization outputs:** aggregate interface P and Q.
- **Inactive actuators:** DER Q, switched shunts, and all transmission-system controls.

## Interpretation

The result characterizes the intrinsic DSO exchange under a stiff EHV
boundary rather than the behavior of the complete IEEE 39 + DSO system. The
earlier low-voltage/non-convergence result was attributable to endogenous,
strongly depressed IEEE 39 primary voltages and should be retained only as a
coupled-system stress diagnostic.
