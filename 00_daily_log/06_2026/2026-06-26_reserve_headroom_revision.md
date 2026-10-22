# 2026-06-26 - Reserve headroom based tie coordination

Timestamp: 2026-06-26 13:41:50 +02:00

## Reason

The previous `--reserve` extension used a normalized reserve-position signal as the economic marginal proxy. This treated a zone with little or no controllable capability as apparently "abundant" when its normalized reserve was low, and it did not distinguish whether a zone had usable capacitive and inductive Q headroom. The resulting experiment could activate the TSO-TSO coordinator but was not a meaningful proof that one zone helps another in a grid-friendly way.

## Change

- Added a TSO-side absolute reserve-headroom report returning capacitive and inductive Q headroom in Mvar from online SG and TSO-DER actuators.
- Added `tie_reserve_headroom_scale_mvar` to convert limiting headroom into a bounded scarcity proxy:
  `mu = H0 / (min(H_cap, H_ind) + H0)`.
- Rewired the multi-TSO runner so reserve/headroom diagnostics are computed for both coordinated and uncoordinated runs.
- Folded slack/equivalent machines into the zone-level reserve diagnostic by electrical zone, using their actual Q operating point and Q limits.
- Extended experiment records with per-zone `H_cap`, `H_ind`, and limiting absolute headroom fields.
- Updated `007_TIE_COORDINATION.py --reserve` to plot and print the limiting absolute headroom, and to use an active tie-flow soft band during the reserve demonstration.

## Validation

Focused tests:

```text
tests/test_tie_coordinator.py
tests/test_tie_coordination_hooks.py
tests/test_tso_output_gradient.py

32 passed in 67.16 s
```

Experiment:

```text
python experiments/007_TIE_COORDINATION.py --reserve
```

The revised diagnostic identifies zone 2 as the limiting zone by absolute Q headroom:

```text
steady zone reserve scarcity (0 abundant .. 1 saturated):
  OFF        : Z1=0.085  Z2=0.607  Z3=0.256
  COORD-sub  : Z1=0.086  Z2=0.613  Z3=0.259
  COORD-econ : Z1=0.085  Z2=0.609  Z3=0.258

steady limiting absolute headroom min(H+,H-) [Mvar]:
  OFF        : Z1=5371.8  Z2=323.6  Z3=1450.6
  COORD-sub  : Z1=5330.3  Z2=315.5  Z3=1432.5
  COORD-econ : Z1=5358.7  Z2=320.8  Z3=1435.7
```

## Interpretation

The revised signal fixes the main wrong premise in the previous experiment: reserve scarcity is now tied to absolute remaining controllable Q headroom, not to a normalized position that can misclassify no-capability zones. However, the current `007 --reserve` scenario still shows only a small economic-coordination effect. This indicates that the remaining limitation is likely experimental: the stressed zone has low local headroom, but the tested tie-voltage actuation has weak deliverability leverage under the present network operating point and constraints.

## Next work

- Add a direction-specific deliverability metric before using `mu_j - mu_i` as a routing target.
- Construct a reserve stress case where one neighboring zone has demonstrably spare Q headroom and a non-negligible sensitivity of the constrained zone reserve/voltage objective to the tie voltage difference.
- Report active tie-flow deviation and voltage-security margins together with reserve equalization, so "helping" is accepted only when it remains grid-friendly.
