# Scenario: `wind_replace`

Replaces selected synchronous generators with STATCOM-capable wind parks.
The replacement scaling differs per zone to create a differentiated
generation mix.

## Removed Generators

| Gen idx | Grid bus (IEEE) | Zone | Original P [MW] | Replacement |
|:-------:|:---------------:|:----:|:---------------:|-------------|
| 0 | 1 (IEEE 2) | 1 | 250 | WP at same P |
| 2 | 18 (IEEE 19) | 3 | 632 | WP at same P |
| 3 | 18 (IEEE 19) | 3 | 508 | WP at same P |
| 6 | 24 (IEEE 25) | 1 | 540 | WP at same P |
| 8 | 5 (IEEE 6) | 2 | 500 | WP at half P (250 MW) |

## Remaining Synchronous Generators

| Gen idx | Terminal bus | Grid bus (IEEE) | Zone | P [MW] | Role |
|:-------:|:-----------:|:---------------:|:----:|-------:|------|
| 1 | 31 | 9 (IEEE 10) | 2 | 650 | Zone 2 anchor (IEEE G3) |
| 4 | 34 | 21 (IEEE 22) | 3 | 650 | Zone 3 anchor |
| 5 | 39 | 35 (IEEE 36) | 3 | 560 | Zone 3 anchor |
| 7 | 37 | 28 (IEEE 29) | 1 | 830 | Zone 1 anchor |

## Slack

| Element | Bus (IEEE) | vm_pu |
|---------|:----------:|:-----:|
| ext_grid 0 | 38 (IEEE 39) | 1.03 |

## STATCOM Wind Parks

Sgen indices are ordered by the sequence in which gens were removed (see
`apply_wind_replace` in `wind_replace.py`); the mapping below assumes the
current loop order (Zone 2 first, then Zones 1 and 3).

| Bus (IEEE) | Zone | P [MW] | S_n [MVA] | Q_avail [Mvar] | Replaced gen |
|:----------:|:----:|-------:|----------:|:--------------:|:------------:|
| 5 (IEEE 6)   | 2 | 250 | 300 | 166 | gen_idx 8 (ex-slack) |
| 1 (IEEE 2)   | 1 | 250 | 300 | 166 | gen_idx 0 (G0) |
| 18 (IEEE 19) | 3 | 632 | 758 | 419 | gen_idx 2 (G2) |
| 18 (IEEE 19) | 3 | 508 | 610 | 337 | gen_idx 3 (G3) |
| 24 (IEEE 25) | 1 | 540 | 648 | 358 | gen_idx 6 (G6) |

Q_avail = sqrt(S_n^2 - P^2) at full active power output.

## Per-Zone Summary

| Zone | Sync gens | STATCOMs | Total P_sync [MW] | Total P_WP [MW] |
|:----:|:---------:|:--------:|:-----------------:|:---------------:|
| 1 | 1 (gen_idx 7) + slack | 2 | 830 + slack | 790 |
| 2 | 1 (gen_idx 1, IEEE G3) | 1 | 650 | 250 |
| 3 | 2 (gen_idx 4, 5) | 2 | 1210 | 1140 |

## Converter Sizing

- `S_n = 1.2 * P_max` (20% oversized for reactive power headroom)
- Operating diagram: `STATCOM` (full-circle PQ capability, no dead zone at P=0)
- Profile: `WP10` (wind power, capacity factor ~28%)

## Q Initialization

STATCOM Q is set in two stages with different roles:

### Stage 1 — seed (inside `apply_wind_replace`)

Immediately after sgen creation, a temporary PV generator is placed at
each STATCOM bus with `vm_pu = ext_grid_vm_pu`.  The PF solver finds
the Q needed to hold that voltage; the Q is copied back to the sgen
and the temp gen is dropped.  This is a **seed only** — its purpose is
to give downstream operations (e.g. `add_hv_networks`, the base PF) a
physically plausible starting point so they converge.  The Q value
written here is expected to be overwritten.

### Stage 2 — authoritative (caller, after profiles)

The caller (e.g. `run_multi_tso_dso` in `experiments/000_M_TSO_M_DSO.py`)
runs a single combined init pass **after** profiles and gen dispatch
are applied:

1. **Phase 1 (TSO side).**  Disable STATCOM sgens, create temporary PV
   generators at each STATCOM bus with `vm_pu = v_setpoint_pu`, and add
   `DiscreteTapControl` on the machine 2W transformers.  A single
   `runpp(run_control=True)` co-converges machine OLTC taps and STATCOM
   Q.  The Q values are transferred back to the sgens and the temporary
   gens + machine-trafo controllers are removed.
2. **Phase 2 (DSO side).**  Add `DiscreteTapControl` on the coupler 3W
   transformers at `oltc_init_v_target_pu` and run a second
   `runpp(run_control=True)` so HV-subnetwork voltages land near target
   without disturbing the TSO-side operating point already settled in
   Phase 1.

## Notes

- Zone 2 retains one synchronous machine (pandapower `gen_idx=1`, IEEE G3
  at terminal bus 31 / grid bus 9, 650 MW) as the Zone-2 anchor. Only the
  ex-slack gen at terminal 30 is replaced by a STATCOM wind park (at half
  the original 500 MW = 250 MW).
- Zone 1 and Zone 3 wind-park replacements match the original P_mw.
- Zone bus sets are defined locally (not imported from zone_partition) to
  avoid fragile coupling. They must match `_FIXED_ZONES_IEEE39`.
