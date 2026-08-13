# Scenario: `rural_700`

The mixed-rural expansion scenario combines:

- the same validated transmission-side transformation as `base_410`; and
- 700 MW integer-valued installed DER capacity in each synthetic DSO.

Per DSO, installed capacity is 460 MW wind and 240 MW PV:

| Plant | HV bus | Rating |
|---|---:|---:|
| Internal wind | 4 | 70 MW |
| Internal wind | 5 | 100 MW |
| Internal wind | 6 | 80 MW |
| Coupling wind | 0, 3, 8 | 70 MW each |
| PV | 3 | 80 MW |
| PV | 4 | 70 MW |
| PV | 5 | 50 MW |
| PV | 7 | 40 MW |

All installed ratings are integer multiples of 10 MW. The exact values are
defined in `network/ieee39/constants.py::DSO_DER_CAPACITY_SCENARIOS`.
