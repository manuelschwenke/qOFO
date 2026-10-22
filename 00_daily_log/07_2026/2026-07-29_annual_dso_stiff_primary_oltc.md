# Annual DSO characterization with stiff primaries and local OLTCs

**Timestamp:** 2026-07-29 15:54:29 +02:00  
**Reason:** Correct the annual DSO P-Q characterization boundary condition. The previous probe retained the uncontrolled IEEE 39 transmission system, causing unrealistically depressed EHV primary voltages and preventing the DSO OLTCs from reaching their 110 kV targets.

## Established diagnostic result

The original `rural_700` coupled-system probe used the full IEEE 39 network and controlled the 110 kV (`mv`) side of each 345/110/20 kV coupling transformer at 1.03 p.u. The 345 kV primaries were not fixed.

At the first converged point (2016-01-01 12:30), measured primary voltages were:

| DSO | Three 345 kV primary voltages [p.u.] | Coupler taps |
|---|---|---|
| DSO 1 | 0.768, 0.758, 0.793 | -13, -13, -13 |
| DSO 2 | 0.838, 0.847, 0.823 | -13, -13, -13 |
| DSO 3 | 0.842, 0.868, 0.859 | -13, -13, -13 |
| DSO 4 | 0.894, 0.888, 0.910 | -11, -12, -10 |

The `DiscreteTapControl` objects therefore worked in the correct direction but saturated at the lower tap limit for most coupling transformers. The low 110 kV voltages were inherited from the depressed transmission-side boundary, not caused by an inactive tap controller.

## Revised annual-probe model

Each of the four DSOs is now solved as an electrically independent pandapower case:

- the transmission system and the other three DSOs are out of service;
- the target DSO retains its three coupling transformers, 110 kV lines, auxiliary ZIP-load branches, loads, and DER;
- each 345 kV primary terminal is supplied by a stiff `ext_grid` at 1.03 p.u. and 0 degrees;
- the three sources have equal `slack_weight = 1/3`;
- `pandapower.runpp(..., distributed_slack=True)` is used;
- every coupling transformer has a `DiscreteTapControl` controlling its 110 kV (`mv`) terminal at 1.03 p.u.;
- fresh and retry power flows use `init="dc"`; chronological warm starts use `init="results"`;
- switched shunts and TS controls are absent.

The four cases must be solved separately. Pandapower's distributed-slack formulation did not solve four disconnected slack islands correctly in one common network.

## DER reactive-power convention

The preferred characterization remains unity power factor:

\[
Q_{\mathrm{DER}} = 0.
\]

Optional non-unity cases use the project's established inductive convention:

\[
Q_{\mathrm{DER}}
=
-|P_{\mathrm{DER}}|
\tan\!\left(\arccos(\cos\varphi)\right),
\qquad
\cos\varphi \in \{0.98, 0.95\}.
\]

Thus negative pandapower `sgen.q_mvar` denotes inductive reactive-power absorption. Capacitive diagnostic runs were rejected after this sign convention was clarified.

## 96-step `rural_700` validation

Boundary conditions:

- installed DER: 700 MW per DSO;
- primary voltage: 1.03 p.u.;
- OLTC secondary reference: 1.03 p.u.;
- DER power factor: 1.0;
- native 15-minute profiles, first 96 samples.

Results:

- 384/384 DSO power flows converged;
- zero retries;
- no DER reactive support was required;
- aggregate DSO voltage range: approximately 0.998–1.078 p.u.;
- no converged sample was outside 0.9–1.1 p.u.

The generated probe data are stored in:

`results/annual_dso_pq_characterization_isolated_rural_700_probe96/`

## Controlled outputs and actuators

- **Controlled output:** local 110 kV voltage at each coupling-transformer `mv` terminal.
- **Actuator:** discrete coupling-transformer tap position.
- **Characterization outputs:** aggregate active and reactive power at the three 345 kV transformer interfaces.
- **Exogenous quantities:** load P/Q and DER P profiles; DER Q is zero for the selected unity-power-factor case.

## Remaining work

Run the complete annual profile with the same isolated, stiff-primary boundary condition and retain the 96-step probe as a regression fixture. The full-year CSV should be written to a boundary-condition-specific result directory to avoid confusion with the earlier coupled IEEE 39 characterization.
