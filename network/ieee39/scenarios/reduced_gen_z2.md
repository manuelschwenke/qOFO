# Scenario: `reduced_gen_z2`

Removes the ex-slack generator at bus 30 (gen 8, grid bus 5, Zone 2).
Zone 2 retains only Gen 1 (bus 31, 650 MW) as the sole synchronous machine.

## Removed Generators

| Gen idx | Grid bus (IEEE) | Zone | P [MW] | Reason |
|:-------:|:---------------:|:----:|-------:|--------|
| 8 | 5 (IEEE 6) | 2 | 500 | Reduce Zone 2 gen capacity |

## Remaining Synchronous Generators

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

## Slack

| Element | Bus (IEEE) | vm_pu |
|---------|:----------:|:-----:|
| ext_grid 0 | 38 (IEEE 39) | 1.03 |

## Per-Zone Summary

| Zone | Sync gens | Total P_gen [MW] | Notes |
|:----:|:---------:|:----------------:|-------|
| 1 | 3 (G0, G6, G7) + slack | 1620 + slack | Ample capacity |
| 2 | 1 (G1) | 650 | Deficit vs ~1600 MW load; tie-line flows |
| 3 | 4 (G2, G3, G4, G5) | 2350 | Ample capacity |

## Notes

- Zone 2 has only one synchronous generator (Gen 1, max 725 MW) against
  ~1600 MW of zone load. The ~950 MW deficit flows through tie lines
  (Line 2: bus 1-2, Line 14: bus 8-38) and is absorbed by other zones.
- The associated machine transformer and 10.5 kV terminal bus are also
  removed by `remove_generators()`.
