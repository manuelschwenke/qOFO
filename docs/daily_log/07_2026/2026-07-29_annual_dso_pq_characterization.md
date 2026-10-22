# Annual synthetic-DS P-Q characterization

**Timestamp:** 2026-07-29 14:31:33 +02:00  
**Reason:** Generate a full-year TS-DS interface P-Q data set for
characterizing the four synthetic 110 kV distribution systems underlying the
IEEE 39-bus network, with exogenous DER P and no DER Q control.

## Short answer

Implemented and executed
`analysis/annual_dso_pq_characterization.py`. The analysis attempted all
35,136 native 15-minute snapshots of leap year 2016 on the fully coupled
IEEE39 + 4 DSO plant. It produced the auditable full table, compact combined
and per-DSO TikZ/PGFPlots CSVs, a summary CSV, a failure audit, run metadata,
and a diagnostic scatter plot in
`results/annual_dso_pq_characterization/`.

The standard AC Newton-Raphson power flow converged for 35,120 snapshots
(99.9545 %) and failed for 16 snapshots (0.0455 %). Failed cases remain in
the raw table with `converged=False` and are excluded from the scatter CSVs.

## Assumptions, constraints, actuators, and controlled outputs

### Assumptions

- The request phrase "Q from load time-series" is interpreted as **no
  distribution-DER Q control**: every synthetic-DS `sgen.q_mvar` is fixed at
  0 Mvar.
- Positive interface P and Q denote import from the TS into a DSO. The
  interface value is the sum of `res_trafo3w.p_hv_mw` or
  `res_trafo3w.q_hv_mvar` over the three interface transformers.
- The complete 2016 native profile index is used (15-minute resolution,
  35,136 samples).
- Four missing HS4/HS5 transmission-load samples per profile and one missing
  `mv_rural_qload` sample were filled. Interior gaps use linear
  interpolation; boundary gaps use the same quarter-hour of the adjacent day.
  Complete DSO DER P profiles (`PV3`, `WP7`, `WP10`) and the DSO active-load
  profile are unchanged.

### Constraints and plant state

- Fully coupled IEEE 39-bus TS plus DSO_1 through DSO_4.
- Constant-PQ load model; no controller is allowed to query or modify the
  plant during the annual sweep.
- Coupler and machine OLTC positions remain at their constructed values.
- TSO tertiary switched shunts are not installed for this passive
  characterization.
- Synchronous generators retain pandapower PV-bus voltage regulation;
  synchronous active-power schedules remain fixed and the designated slack
  balances the system.
- Newton-Raphson uses voltage angles, no distributed slack, no generator-Q
  limit enforcement, and warm starts/recycled bus injections after the first
  snapshot.

### Actuators

- DSO DER P is exogenous through `PV3`, `WP7`, and `WP10`.
- DSO DER Q is fixed at 0 Mvar.
- OLTCs and switched shunts are fixed/inactive.
- No MIQP/OFO, local Q(V), or cos(phi) controller is executed.

### Controlled outputs

None. This is an uncontrolled operating-point characterization. Recorded
outputs are aggregate and per-transformer interface P/Q, DS load/DER/net
demand, passive-network P/Q differences, DS voltage extrema, and line/coupler
loading.

## Method and code change

- Added a vectorized profile applicator that is equivalent to
  `core.profiles.apply_profiles` for the participating load and `sgen` rows.
- Added restart checkpoints so long annual sweeps can resume safely.
- Added warm-started/recycled AC power flows with a fresh-conversion retry.
- Added explicit failure retention instead of silently dropping time steps.
- Added:
  - `annual_pq_timeseries.csv`: 140,544 rows (all time/DSO combinations);
  - `annual_pq_scatter.csv`: 140,480 converged time/DSO points;
  - `pq_scatter_DSO_1.csv` through `pq_scatter_DSO_4.csv`: two-column
    `p_mw,q_mvar` PGFPlots inputs;
  - `annual_pq_summary.csv`;
  - `annual_pf_failures.csv`;
  - `annual_pq_characterization.png`;
  - `run_metadata.json` and output `README.md`.

## Main results

| DSO | P range [MW] | Q range [Mvar] | P mean [MW] | export fraction | annual Vmin [pu] | samples outside 0.9–1.1 pu |
|---|---:|---:|---:|---:|---:|---:|
| DSO_1 | -154.61 to 243.82 | 71.63 to 107.83 | 73.89 | 18.86 % | 0.7941 | 1,971 |
| DSO_2 | -152.61 to 245.23 | 65.68 to 104.82 | 74.43 | 18.75 % | 0.8483 | 888 |
| DSO_3 | -155.94 to 243.30 | 71.93 to 101.22 | 73.52 | 18.93 % | 0.8899 | 15 |
| DSO_4 | -149.66 to 247.40 | 58.30 to 103.04 | 75.43 | 18.51 % | 0.9182 | 0 |

Every DSO has 410 MW installed DER and a 261.80375 MW reference active load.
Differences among the four interface clouds therefore arise from electrical
topology/line-length scale, interface voltage, and losses, not different
installed P profiles.

## Verification

- CSV integrity: 35,136 timestamps, four DSOs, 140,544 raw rows, and 140,480
  converged scatter rows.
- Artifact-tool import check: combined scatter header is
  `timestamp,dso_id,p_mw,q_mvar`; sampled first/last records parse as a
  four-column table.
- `max(abs(q_der_mvar)) = 0`.
- Aggregate interface P/Q reproduce the sum of the three coupler values.
- P and Q decomposition residuals are at CSV rounding precision
  (approximately `1e-9` MW/Mvar).
- The 16 failed timestamps were also tried separately with NR auto/flat/DC,
  BFSW, FDBX, FDXB, and Gauss-Seidel; none converged. They are therefore
  treated as non-solutions of the fixed-tap, no-DER-Q operating mode rather
  than replaced by fabricated P-Q values.

## Risks and unresolved points

- The scatter is not a feasible P-Q **capability region**. It is the
  trajectory produced by one year of correlated exogenous P/load profiles.
- Interface Q is not exactly the arithmetic load-Q sum: line charging,
  transformer magnetizing/series exchange, losses, and interface voltage
  contribute. Both `q_net_demand_mvar` and `q_passive_network_mvar` are
  retained to make this distinction explicit.
- The large undervoltage count for DSO_1 and DSO_2 and the 16 non-solutions
  show that fixed neutral taps plus Q=0 DER is not an admissible operating
  policy over the complete profile year. A separate comparison with OLTC
  regulation (still keeping DER Q=0) would quantify the transformer-control
  contribution without conflating it with DER Q control.
- `enforce_q_lims=False` means synchronous TS generators retain ideal PV-bus
  voltage control even if their computed Q exceeds a physical envelope.

## Suggested next Obsidian note

Create `2026-07-29_annual_dso_pq_fixed_oltc_vs_regulated_oltc.md` and compare:

1. the present fixed-tap, DER-Q=0 baseline;
2. regulated interface OLTCs with DER-Q still fixed at zero; and
3. the later coordinated DER-Q case.

Report convergence, voltage admissibility, interface-P/Q cloud displacement,
and passive-network Q separately. This would turn the present data set into a
clean actuator-ablation study.
