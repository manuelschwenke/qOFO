# DSO generation-capacity scenarios

**Timestamp:** 2026-07-29 15:17:07 +02:00  
**Reason:** Replace the former `wind_replace` / `reduced_gen_z2` public scenario selection with two explicit synthetic DSO generation-capacity cases: the established 410 MW case and a mixed-rural 700 MW case.

## Changes

- Added `base_410` and `rural_700` as the only public IEEE 39 pandapower scenario names.
- Both cases retain the validated `wind_replace` transmission-system transformation as their common TS basis.
- Removed the obsolete `reduced_gen_z2` scenario module and documentation.
- Added capacity tables in `network/ieee39/constants.py` and made `add_hv_networks()` select the appropriate table from the IEEE 39 scenario stored on the network.
- Changed the default configuration and active experiment/test callers from `wind_replace` to `base_410`.
- Kept `wind_replace` as a deprecated input alias for `base_410` to avoid silently breaking old configuration files. It is not part of the public scenario registry.
- Added scenario documentation and regression tests.
- Made the annual DSO P-Q characterization script scenario-aware and separated the default output directory for `rural_700`.

## Installed capacity per synthetic DSO

### `base_410`

| Technology / connection | Installed capacity |
|---|---:|
| Wind, internal HV buses 4 / 5 / 6 | 40 / 60 / 50 MW |
| Wind, three coupling buses | 40 / 40 / 40 MW |
| PV, internal HV buses 3 / 4 / 5 / 7 | 50 / 40 / 30 / 20 MW |
| **Wind total** | **270 MW** |
| **PV total** | **140 MW** |
| **Total** | **410 MW** |

### `rural_700`

| Technology / connection | Installed capacity |
|---|---:|
| Wind, internal HV buses 4 / 5 / 6 | 70 / 100 / 80 MW |
| Wind, three coupling buses | 70 / 70 / 70 MW |
| PV, internal HV buses 3 / 4 / 5 / 7 | 80 / 70 / 50 / 40 MW |
| **Wind total** | **460 MW** |
| **PV total** | **240 MW** |
| **Total** | **700 MW** |

All specified capacities are integer multiples of 10 MW.

## Assumptions and controller interpretation

- The capacity total applies to each synthetic DSO underlying an IEEE 39 interface.
- The TS topology and TS generation replacement are identical in both cases; only installed wind/PV capacity in the underlying HV network changes.
- The annual P-Q characterization uses active-power time series for loads and DER. Under the passive reference policy, DER reactive power is fixed to zero and load reactive power follows the load time series.
- In the hierarchical-control experiments, DSO DER reactive power and OLTC taps remain available actuators. The controlled DSO output is reactive power at the EHV-HV interface, subject to nodal-voltage and equipment constraints.

## Validation

- Targeted scenario and DER-tagging test suite: **18 passed**.
- Both `base_410` and `rural_700` networks converge at their constructed nominal operating point.
- A 96-step annual-profile smoke test for `rural_700` correctly reported 700 MW installed capacity per DSO.

## Risk / open result

With fixed taps and `Q_DER = 0`, only 46 of the first 96 annual-profile operating points converged for `rural_700`; 50 failed, and the lowest observed voltage among converged points was approximately 0.737 p.u. This does not invalidate the installed-capacity scenario, but it shows that 700 MW is not generally feasible as an uncontrolled passive operating policy with the present time-series scaling and voltage constraints. A full-year data set intended to characterize the controlled DSO should therefore include the intended OLTC and/or DER-Q control policy, or explicitly retain convergence status so that infeasible points are not mistaken for missing data.
