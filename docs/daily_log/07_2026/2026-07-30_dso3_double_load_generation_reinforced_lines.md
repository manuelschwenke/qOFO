# DSO 3 doubled load/generation probe and reinforced HV conductors

Date: 2026-07-30

> Historical intermediate result. Superseded later on 2026-07-30 by
> `2026-07-30_ieee39_dso_powerfactory_sync_handover.md`: all DSOs now use
> profile-only 500 Mvar Q, and DSO 3 corridor (5-6) has two circuits.

## Short answer

The annual isolated-DSO characterization can now apply auditable per-DSO
load, DER, and optional line-type overrides. The requested DSO 3 stress case
uses 1,400 MW installed DER, twice its active-load profile bases
(523.6075 MW nominal/reference load), and 50 Mvar nominal/reference reactive
load. The physical IEEE39 underlay now uses pandapower standard conductors:

- DSO 1, DSO 2, DSO 4: 305-AL1/39-ST1A 110.0 (0.740 kA).
- DSO 3: 490-AL1/64-ST1A 110.0 (0.960 kA).

The full 2016 run solved all 140,544 power flows without retries. DSO 3
nevertheless reaches 121.17% internal line loading during the strongest
summer backfeed, so conductor diameter alone does not remove every thermal
constraint.

## Reason and scope

Non-public mixed-rural systems show approximately twice the load and
backfeed of the original synthetic systems. Only DSO 3 was scaled to
represent this larger system. The other three DSOs retain rural_700 load and
DER quantities but receive the requested 305 mm2 conductor.

The two approved generation scenarios (base_410 and rural_700) remain
unchanged. DSO 3 quantity scaling is an analysis override on rural_700;
conductor types are a persistent physical topology fact in the IEEE39/HV
network constructor.

## Assumptions, constraints, actuators, and outputs

- Boundary: three stiff 345 kV sources per DSO at 1.03 pu with equal
  distributed-slack weights; transmission system excluded.
- OLTC actuator: three pandapower DiscreteTapControl instances per DSO,
  regulating 110 kV terminals to 1.03 pu.
- DER P: exogenous PV3, WP7, and WP10 time series.
- DER Q: unity power factor (Q_DER = 0).
- Load P/Q: native 15-minute time series; DSO 3 P bases multiplied by 2.0
  and nominal Q reference changed from 80 to 50 Mvar.
- Switched shunts: inactive.
- Controlled output: local coupling-bus voltage through OLTC action.
- Characterization outputs: primary-side interface P/Q, voltage, loading,
  losses, and tap positions. Positive interface P/Q denotes TS-to-DS import.

Active-load profile rows are peak/mean normalized, so their stored base_p_mw
rows do not sum to nominal/reference load. The factor-of-two override
multiplies those profile bases, while HVNetworkInfo.total_ref_p_mw changes
from 261.80375 to 523.6075 MW. For Q, the 50 Mvar nominal constant component
retains a 25 Mvar profile-swing base; DSO 3 base_q_mvar rows sum to 75 Mvar.

## Code changes

- network/ieee39/constants.py: added DSO_HV_LINE_STD_TYPES.
- network/ieee39/hv_networks.py: selects the conductor by DSO and imports
  omitted types from pandapower's built-in standard library.
- analysis/annual_dso_pq_characterization.py: added repeatable per-DSO CLI
  overrides for DER scale, active-load scale, nominal Q base, and optional
  line type; records overrides and physical line types in metadata/README;
  adds nominal Q reference to the summary.
- tests/test_annual_dso_pq_characterization.py: verifies DSO 3 scaling and
  the 305/490 mm2 conductor assignment.

Focused validation: 30 tests passed. One unrelated deprecation warning
remains for the wind_replace alias in export/make_snapshots.py.

## Results

January reinforced probe:

- 11,904 / 11,904 converged; no retries.
- DSO 3 P: -614.25 to +482.54 MW; Q: +43.63 to +125.51 Mvar.
- DSO 3 voltage: 1.0202 to 1.0427 pu.
- DSO 3 maximum line/coupler loading: 81.16% / 68.05%.

Full 2016 reinforced run:

- 140,544 / 140,544 converged; no retries.
- DSO 3 P: -791.47 to +482.54 MW; Q: +43.37 to +211.02 Mvar.
- DSO 3 voltage: 1.0178 to 1.0427 pu; no samples outside 0.9--1.1 pu.
- DSO 3 maximum line/coupler loading: 121.17% / 89.15%.

The annual limiting case occurs on 2016-07-25 13:00 at DSO_3|Line_(5-6).
A fresh-DC diagnostic gives about 120.7% there and 101.6% on line (3-4).
A second 490 mm2 circuit on line (5-6) alone reduces the tested maximum to
about 97.9%, with line (1-2) then limiting. This circuit is not implemented
pending architectural agreement.

## Reproducible output

- January: results/annual_dso_pq_characterization_isolated_rural_700_dso3_x2_january_reinforced/
- Full year: results/annual_dso_pq_characterization_isolated_rural_700_dso3_x2_reinforced/

The full-year folder contains the time series, per-DSO TikZ CSVs, summary,
failures table, metadata, README, and PNG plot.

## Risks / unresolved points

1. Decide whether the full-year 121% DSO 3 line constraint is intentional or
   whether line (5-6) should receive a second 490 mm2 circuit.
2. The DSO 3 P/Q envelope is nonlinear because losses, charging, OLTC taps,
   and voltages change with the operating point.
3. Do not replace thesis data before deciding whether the residual overload
   is an intended capability constraint or an unacceptable base-case state.
