# DSO 3 profile-only reactive-load stress probe

Date: 2026-07-30

> Historical intermediate result. Superseded later on 2026-07-30 by
> `2026-07-30_ieee39_dso_powerfactory_sync_handover.md`: the profile-only
> 500 Mvar Q model now applies to all four DSOs, not only DSO 3.

## Short answer

DSO 3 can now be characterized without any constant reactive-load component.
Its aggregate profiled reactive-load base was set to 500 Mvar, ten times the
previous requested 50 Mvar reference. Three representative snapshots and the
96 quarter-hours of 2016-07-25 converged at unity DER power factor without
retries.

## Mathematical definition

For DSO 3:

    Q_load(t) = 500 Mvar * mv_rural_qload(t)

All load rows without a profile_q assignment have q_mvar = base_q_mvar = 0.
The ten profile-controlled rows sum to base_q_mvar = 500 Mvar.

Existing spatial weights are retained:

- HV buses 0--5: raw weight 1, normalized weight 10/14.
- HV buses 6--9: raw weight 2, normalized weight 20/14.
- Base per low-load bus: 500/10 * 10/14 = 35.714 Mvar.
- Base per high-load bus: 500/10 * 20/14 = 71.429 Mvar.

The weighted aggregate is
6 * 35.714 + 4 * 71.429 = 500 Mvar. Each bus value is multiplied by the same
mv_rural_qload sample.

## Profile facts

The 2016 mv_rural_qload series is signed:

- Minimum: -0.139967 at 2016-05-15 04:30.
- Mean: -0.049883.
- Median: -0.060527.
- Maximum: +0.344266 at 2016-07-25 13:00.

The 500 Mvar base therefore gives aggregate DSO 3 load Q from -69.98 to
+172.13 Mvar and mean -24.94 Mvar. Negative values are capacitive injection
under pandapower's load sign convention. This is an unavoidable consequence
of removing the constant reference while preserving the signed source series.

## Assumptions and controls

- DSO 3 DER nameplate 1,400 MW and active-load bases scaled by 2.
- DSO 3 line type 490-AL1/64-ST1A 110.0; other DSOs use 305 mm2.
- Three stiff 345 kV sources per DSO at 1.03 pu with distributed slack.
- Three DiscreteTapControl OLTCs per DSO regulate 110 kV to 1.03 pu.
- DER reactive power is zero (unity power factor).
- Switched shunts are inactive.
- Interface P/Q are uncontrolled characterization outputs at the primary
  sides of the three coupling transformers.

## Implementation

analysis/annual_dso_pq_characterization.py now provides:

    --dso-load-q-profile-base-mvar DSO_3=500

This mode zeros constant-Q rows and scales only profile_q rows. It cannot be
combined with --dso-load-q-base-mvar for the same DSO. Run metadata records
the selected mode and aggregate base.

tests/test_annual_dso_pq_characterization.py verifies that DSO 3 constant-Q
rows are zero, profiled bases sum to 500 Mvar, and the other DSOs remain at
their original 80 Mvar constant plus 40 Mvar profile-swing bases.

Focused validation: 5 tests passed.

## Probe results

- Q-profile minimum snapshot: 4/4 converged; DSO 3 Q_load = -69.98 Mvar,
  interface Q = -62.25 Mvar, voltage 1.027--1.033 pu.
- Q-profile maximum snapshot: 4/4 converged; DSO 3 Q_load = +172.13 Mvar,
  interface Q = +331.13 Mvar, voltage 1.015--1.031 pu, line loading 121.66%,
  coupling-transformer loading 96.67%.
- Reference snapshot: 4/4 converged; DSO 3 interface Q = +30.54 Mvar.
- 2016-07-25 chronological probe: 384/384 power flows converged, no retries.
  DSO 3 interface Q ranged from -25.99 to +343.29 Mvar, voltage from 0.977
  to 1.049 pu, line loading reached 125.72%, and coupling loading 95.71%.

Output:
results/dso3_q_profile_only_500_2016-07-25_probe/

## Risks / unresolved points

1. The signed profile produces capacitive load-Q periods. Confirm that this is
   intended before a month or annual run.
2. The high-Q day remains numerically convergent but violates an internal line
   rating; it is unsuitable as an unconstrained base case without reinforcement.
3. The coupling transformers approach their thermal rating at maximum Q.
