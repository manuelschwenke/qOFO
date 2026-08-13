# Scenario: `base_410`

The public baseline scenario combines:

- the validated transmission-side transformation implemented by
  `apply_wind_replace`; and
- 410 MW integer-valued installed DER capacity in each synthetic DSO.

Per DSO, installed capacity is 270 MW wind and 140 MW PV. The wind total
contains three 40 MW coupling-bus wind parks.

The exact plant ratings are defined in
`network/ieee39/constants.py::DSO_DER_CAPACITY_SCENARIOS`.
