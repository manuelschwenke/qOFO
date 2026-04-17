# Scenario: `wind_replace`

Replaces selected synchronous generators with STATCOM-capable wind parks.
The replacement scaling differs per zone to create a differentiated
generation mix.

## Removed Generators

| Gen idx | Grid bus (IEEE) | Zone | Original P [MW] | Replacement |
|:-------:|:---------------:|:----:|:---------------:|-------------|
| 0 | 1 (IEEE 2) | 1 | 250 | WP at same P |
| 1 | 9 (IEEE 10) | 2 | 650 | WP at half P (325 MW) |
| 2 | 18 (IEEE 19) | 3 | 632 | WP at same P |
| 3 | 18 (IEEE 19) | 3 | 508 | WP at same P |
| 6 | 24 (IEEE 25) | 1 | 540 | WP at same P |
| 8 | 5 (IEEE 6) | 2 | 500 | WP at half P (250 MW) |

## Remaining Synchronous Generators

| Gen idx | Terminal bus | Grid bus (IEEE) | Zone | P [MW] | Role |
|:-------:|:-----------:|:---------------:|:----:|-------:|------|
| 4 | 34 | 21 (IEEE 22) | 3 | 650 | Zone 3 anchor |
| 5 | 39 | 35 (IEEE 36) | 3 | 560 | Zone 3 anchor |
| 7 | 37 | 28 (IEEE 29) | 1 | 830 | Zone 1 anchor |

## Slack

| Element | Bus (IEEE) | vm_pu |
|---------|:----------:|:-----:|
| ext_grid 0 | 38 (IEEE 39) | 1.03 |

## STATCOM Wind Parks

| Sgen idx | Bus (IEEE) | Zone | P [MW] | S_n [MVA] | Q_avail [Mvar] | Replaced gen |
|:--------:|:----------:|:----:|-------:|----------:|:--------------:|:------------:|
| 0 | 1 (IEEE 2) | 1 | 250 | 300 | 166 | G0 |
| 1 | 9 (IEEE 10) | 2 | 325 | 390 | 216 | G1 |
| 2 | 18 (IEEE 19) | 3 | 632 | 758 | 419 | G2 |
| 3 | 18 (IEEE 19) | 3 | 508 | 610 | 337 | G3 |
| 4 | 24 (IEEE 25) | 1 | 540 | 648 | 358 | G6 |
| 5 | 5 (IEEE 6) | 2 | 250 | 300 | 166 | G8 |

Q_avail = sqrt(S_n^2 - P^2) at full active power output.

## Per-Zone Summary

| Zone | Sync gens | STATCOMs | Total P_sync [MW] | Total P_WP [MW] |
|:----:|:---------:|:--------:|:-----------------:|:---------------:|
| 1 | 1 (G7) + slack | 2 (sgen 0, 4) | 830 + slack | 790 |
| 2 | 0 | 2 (sgen 1, 5) | 0 | 575 |
| 3 | 2 (G4, G5) | 2 (sgen 2, 3) | 1210 | 1140 |

## Converter Sizing

- `S_n = 1.2 * P_max` (20% oversized for reactive power headroom)
- Operating diagram: `STATCOM` (full-circle PQ capability, no dead zone at P=0)
- Profile: `WP10` (wind power, capacity factor ~28%)

## Q Initialization

STATCOM Q is initialized via temporary PV generators at each bus with
`vm_pu = 1.03`. The PF solver finds the Q injection needed to maintain
that voltage. The Q values are transferred to the sgens, then the
temporary generators are removed.

This runs twice:
1. In `build_ieee39_net()` — after generator replacement (base operating point)
2. In `run_multi_tso_dso()` — after profiles and gen dispatch are applied
   (profile-scaled operating point, before OLTC initialization)

## Notes

- Zone 2 has NO synchronous generators. All active power and Q support
  comes from STATCOMs. The slack absorbs Zone 2 imbalances via tie lines.
- Zone 2 wind parks are at HALF the original generator P_mw. Zones 1 and 3
  replacements match the original P_mw.
- Zone bus sets are defined locally (not imported from zone_partition) to
  avoid fragile coupling. They must match `_FIXED_ZONES_IEEE39`.
