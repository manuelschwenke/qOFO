# Scenario: `base`

All 9 PV generators and the slack generator remain active.
No modifications to the IEEE 39-bus New England network.

## Synchronous Generators

| Gen idx | Terminal bus | Grid bus (IEEE) | Zone | P [MW] |
|:-------:|:-----------:|:---------------:|:----:|-------:|
| 0 | 29 | 1 (IEEE 2)  | 1 | 250 |
| 1 | 31 | 9 (IEEE 10) | 2 | 650 |
| 2 | 32 | 18 (IEEE 19) | 3 | 632 |
| 3 | 33 | 18 (IEEE 19) | 3 | 508 |
| 4 | 34 | 21 (IEEE 22) | 3 | 650 |
| 5 | 39 | 35 (IEEE 36) | 3 | 560 |
| 6 | 36 | 24 (IEEE 25) | 1 | 540 |
| 7 | 37 | 28 (IEEE 29) | 1 | 830 |
| 8 | 30 | 5 (IEEE 6)  | 2 | 500 |

## Slack

| Element | Bus (IEEE) | vm_pu |
|---------|:----------:|:-----:|
| ext_grid 0 | 38 (IEEE 39) | 1.03 |

## Notes

- Gen 8 at grid bus 5 was created by `swap_slack_to_bus38()` — the original
  ext_grid at bus 30 was moved to bus 38, and a new PV gen replaced it.
- Machine trafos with index -1 in `machine_trafo_gen_map` are network OLTCs
  at bus 12 (not generator step-ups).
