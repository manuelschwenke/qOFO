# Scenario: `rural_700_north_res` (default since 2026-11-08)

The study network of every result from 2026-11-08 on. Selected by the scarcity screen
(`experiments/ch_08_setup/network_scarcity/`, daily log
`docs/daily_log/11_2026/2026-11-08b_network_scarcity_screen.md`).

## Against `rural_700` (the network of every earlier result)

| | `rural_700` | `rural_700_north_res` |
|---|---|---|
| Synchronous fleet | G1 (equivalent), G3, G4, G7, G9, G10 | unchanged |
| TS wind parks | W1 540 MW @ IEEE 25 (z1), W2 500 @ 6 (z2), W3 508 @ 19 (z3), W4 650 @ 22 (z3) | W3 508 @ 19 (z3) and **one 1500 MW park @ IEEE 26 (z1)** |
| Zone 2 transmission IBG | W2 | **none** |
| STS IBG | 700 MW per DSO (DSO_3 ×2), VDE-AR-N 4120 v2 | unchanged, except: |
| Coupling-bus parks (3 × 70 MVA per STS, ×2 in DSO_3) | 4120 v2 (no Q below P/S_n = 0.1) | **reserved capability**: [−0.33, +0.41]·S_n at every P |
| G1 reactive limit | ±1200 Mvar (config) | unchanged (needs ≤ 531 Mvar in W1–W4, W6) |

The 1500 MW park: profile WP10, S_n = P_n, VDE-AR-N 4120 v2, TS IBG of zone 1. The
machines G8, G2, G5, G6 are removed as before; their parks W1, W2, W4 are not built.

Code: `wind_replace.py` (`ts_wind_layout="north"`, `NORTH_WIND_BUS`, `NORTH_WIND_MW`),
`constants.py` (`coupling_wp_op_diagram`, `RESERVED_OP_DIAGRAM`), `core/actuator_bounds.py`
and `controller/der_qv_local_loop.py` (the reserved diagram).

## Screen evidence (376 MC operating points, LOC initialisation)

- Initialisation 98.4 % (base 96.8 %); N-1 (frozen IBG droops) no new failing element;
  345 kV line loading p95 129 % (base 130 %; ratings not enforced in either).
- Post-contingency V_min p5 0.8994 pu (base 0.9055); accepted by the author after the result.
- Area 2 cannot absorb +400 Mvar of extra demand with its own resources in 15.4 % of the OPs
  (base 4.1 %); almost all of them are hours with G3 decommitted (G3 off-line 15 %, base 22 %).
- Area-2 STS upward capability p10: 344 Mvar (base 0 under 4120 v2).
- G1 operability rule (LOC, W1–W4, W6, 12 h): operable at every limit down to ±400 Mvar;
  G1 needs 347 (W1) / 531 (W6) Mvar, base 1035 / 1259 (the base fails W6 at ±1200).

## Not built here (screen candidates only)

`rural_700_res`, `rural_700_north`, `rural_700_res_no939`, `rural_700_north_res_no939`
stay registered so the screen runs are reproducible. Line 9–39 not built failed initialisation,
N-1 and thermal criteria.
