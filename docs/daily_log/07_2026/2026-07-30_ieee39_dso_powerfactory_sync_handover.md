# IEEE 39 DSO update and PowerFactory synchronization handover

Date: 2026-07-30

## Short answer

This is the authoritative handover for synchronizing the current pandapower
IEEE 39 distribution underlays to PowerFactory. It supersedes the two earlier
2026-07-30 DSO 3 probe notes.

Persistent physical changes:

1. DSO 1, 2, and 4 use `305-AL1/39-ST1A 110.0`; DSO 3 uses
   `490-AL1/64-ST1A 110.0`.
2. DSO 3 corridor HV bus 5--6 has two identical 490 mm2 circuits in parallel.
   Every other DSO corridor has one circuit.
3. Every DSO has zero constant reactive load and uses
   `Q_load(t) = 500 Mvar * mv_rural_qload(t)`.

Experiment-specific change:

4. In the `rural_700` characterization, only DSO 3 has 2x installed DER and
   2x active-load bases. This doubling is not in `constants.py`; it is applied
   by `_apply_dso_overrides()` in
   `analysis/annual_dso_pq_characterization.py` when the two CLI flags shown
   below are supplied.

## Exact location of the DSO 3 doubling

Run command:

```powershell
python -m analysis.annual_dso_pq_characterization `
  --scenario rural_700 `
  --dso-der-scale DSO_3=2 `
  --dso-load-p-scale DSO_3=2
```

Implementation in `analysis/annual_dso_pq_characterization.py`:

- DER: the rows in `hv.sgen_indices` have `p_mw`, `base_p_mw`, and `sn_mva`
  multiplied by 2.
- Active load: the rows in `hv.load_indices` have `p_mw` and `base_p_mw`
  multiplied by 2; `hv.total_ref_p_mw` is also multiplied by 2.
- Load `sn_mva` is multiplied by the largest applied load scale.
- Reactive load is not doubled. Its physical profile base remains 500 Mvar.

Consequences for DSO 3:

- Installed DER: 700 MW -> 1,400 MW.
- Active-load reference: 261.80375 MW -> 523.60750 MW.
- The `base_p_mw` rows remain profile-normalization coefficients and therefore
  do not sum to the reference load.

For a PowerFactory implementation, keep this as an experiment/scenario
multiplier unless DSO 3 is deliberately being made the new global base model.

## DER inventory

The `rural_700` scenario before the DSO 3 multiplier contains, per DSO:

| Element | HV bus | Profile | Rating / active-power base |
|---|---:|---|---:|
| Wind park | 4 | WP7 | 70 MW/MVA |
| Wind park | 5 | WP10 | 100 MW/MVA |
| Wind park | 6 | WP7 | 80 MW/MVA |
| PV | 3 | PV3 | 80 MW/MVA |
| PV | 4 | PV3 | 70 MW/MVA |
| PV | 5 | PV3 | 50 MW/MVA |
| PV | 7 | PV3 | 40 MW/MVA |
| Coupling wind park | 0 | WP10 | 70 MW/MVA |
| Coupling wind park | 3 | WP10 | 70 MW/MVA |
| Coupling wind park | 8 | WP10 | 70 MW/MVA |

Total: 700 MW/MVA. Apply factor 2 to every listed DSO 3 element for the
characterization case. Annual probes used unity power factor, so
`Q_DER = 0`. The code also supports 0.98 or 0.95 inductive power factor,
implemented as negative pandapower generator `q_mvar`; neither was needed.

## Load model to reproduce

All four DSOs:

- Ten HV load locations, buses 0--9.
- Raw spatial weight 1 on buses 0--5 and 2 on buses 6--9.
- Normalized weights are 10/14 and 20/14, respectively.
- Active-load reference is 261.80375 MW per DSO before any override.
- Active power uses a 0.4 constant row plus a 0.6
  `mv_rural_pload / 0.3841` row (`LOAD_PEAK_BOOST = 1.0`).
- Constant-Q rows have `q_mvar = base_q_mvar = 0`.
- Profile-Q rows use `mv_rural_qload` with aggregate base 500 Mvar:
  35.7142857 Mvar at each bus 0--5 and 71.4285714 Mvar at each bus 6--9.

Thus, for every DSO,

```text
Q_load(t) = 500 * mv_rural_qload(t) Mvar.
```

The signed 2016 Q profile has minimum -0.139967, mean -0.049883, and maximum
+0.344266. Therefore each DSO's aggregate load Q ranges from -69.98 to
+172.13 Mvar and averages -24.94 Mvar. Negative load Q is capacitive
injection in pandapower. This signed behavior must be copied intentionally.

## Lines, topology, and couplings

Common unscaled topology `(from, to, km)`:

```text
(0,1,15), (1,2,25), (2,3,20), (3,4,30), (4,5,40), (5,6,30),
(2,6,20), (6,7,15), (7,8,10), (8,9,20), (6,9,15)
```

DSO mapping and length scale:

| DSO | IEEE TN buses -> HV buses | Scale | Conductor |
|---|---|---:|---|
| DSO 1 | (7,8,5) -> (0,3,8) | 0.82 | 305-AL1/39-ST1A 110.0 |
| DSO 2 | (12,14,4) -> (0,3,8) | 1.40 | 305-AL1/39-ST1A 110.0 |
| DSO 3 | (11,10,13) -> (0,3,8) | 0.52 | 490-AL1/64-ST1A 110.0 |
| DSO 4 | (24,21,23) -> (0,3,8) | 2.44 | 305-AL1/39-ST1A 110.0 |

Pandapower standard-type parameters:

| Type | r [ohm/km] | x [ohm/km] | c [nF/km] | max current [kA] |
|---|---:|---:|---:|---:|
| 305-AL1/39-ST1A 110.0 | 0.0949 | 0.380 | 9.20 | 0.740 |
| 490-AL1/64-ST1A 110.0 | 0.0590 | 0.370 | 9.75 | 0.960 |

For DSO 3, line 5--6 is 30 km * 0.52 = 15.6 km and has two identical
parallel circuits. In pandapower this is one line row with `parallel=2`; in
PowerFactory it should be represented by two equal in-service circuits or an
exact equivalent that preserves both series impedance and total ampacity.

Each coupling is a 345/110/20 kV three-winding transformer:

- ratings 300/300/75 MVA;
- uk 12/8/10%, ur 0.30/0.20/0.25%;
- no-load loss 80 kW, no-load current 0.04%;
- MV/LV phase shifts 0/150 degrees;
- ratio tap changer on HV side, neutral 0, range -13...+13, step 1.25%.

## Standalone annual-probe boundary and controls

- Each isolated DSO has three stiff 345 kV sources at 1.03 pu and 0 degrees.
- Distributed slack weights are equal, 1/3 each.
- Three discrete OLTC controllers regulate the 110 kV terminals to 1.03 pu.
- Switched shunts are absent/inactive.
- Interface P and Q are uncontrolled characterization outputs on the
  transformers' 345 kV sides. Positive P/Q denotes TS-to-DS import.
- The stiff-source condition is for isolated DSO characterization only. Do
  not impose it when the DSO is connected to the dynamic transmission model.

## Validation status

Focused construction/analysis tests: 13 passed.

July 2016 probe:

- 11,904/11,904 power flows converged; zero retries.
- No DSO voltage samples outside 0.9--1.1 pu.
- DSO 3: P -792.47...+401.62 MW, Q -70.97...+327.20 Mvar,
  voltage 0.98094...1.04817 pu, coupler loading max 94.83%.
- DSO 3 chronological internal line loading max 102.97%. A direct replay
  identifies line 1--2 as the new limiting path; line 5--6 is no longer the
  limiting corridor after reinforcement.
- DSO 4 has the lowest voltage, 0.92381 pu, still above 0.9 pu.

Full-year output is written to
`results/annual_dso_pq_rural700_allq500_dso3x2_2016/`.

## Risks / unresolved points

1. Confirm that the signed Q profile and its capacitive periods are intended
   in PowerFactory; no constant Q offset remains.
2. The small DSO 3 residual overload shifts to line 1--2. Do not add another
   circuit without a separate architectural decision.
3. Preserve the distinction between persistent plant data (Q model,
   conductors, parallel circuit) and experiment-only DSO 3 P/DER scaling.
4. Compare pandapower and PowerFactory interface P/Q, taps, bus voltages, and
   line loading at common timestamps before accepting parity.
