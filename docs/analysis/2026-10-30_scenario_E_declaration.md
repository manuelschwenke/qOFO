# Declaration: Case-2 scenario E, reactive deficit beyond area 2's own TS resources (written before any run)

Date: 2026-10-30, 23:55, before the first run of scenario E.
Decision: author, 30 Oct evening. The author asked for a scenario with Q scarcity that states
when STS access helps very much, and suggested the outage of 9–39. The design below was approved
("yes please").
Basis: weekend plan §9 (`docs/analysis/2026-10-30_weekend_plan.md` in the `case2-ref` worktree).

## Why a further scenario

- A, B and D gave OFO+ − OFO = −0.39 / −0.47 / −0.43 mpu in aggregate e_v (6–10 %).
- The REF logs show that B and D were **not** Q-scarce for OFO:
  - after the load event OFO held the area-2 mean within +0.6 mpu of 1.03 pu;
  - it used the area-2 TS-DER (120–160 of 205 Mvar) and level 1 of the three MSCs.
- The B/D criterion measured how severe the disturbance was, not whether area 2 could compensate
  it.
- E tests the mechanism hypothesis formed from these results: access pays off strongly once an
  area's reactive deficit exceeds its own TS resources.
  - It is a declared stress test of that condition, not a typical operating case.
  - A stays the reference scenario of ch:case3.

## Schedule

| time | event | element |
|---|---|---|
| 30:40 | G3 out (area 2's only machine, with its machine transformer) | gen 1 |
| 121:40 | line 9–39 out (area 2's tie to the G1 equivalent, 10 GVA) | line 14 |
| 212:20 | 600 Mvar connected at IEEE 4 (location rule, weekend plan §2.4) | load bus 3 |

- The offsets are those of A, B and D, strictly between two TS dispatches.
- All events persist to the end of the 360-min horizon.
- Same noise seed (20261027), period, weights (c08b1d3d33ab) and device limits as the weekend REF
  runs.
- All six variants (LOC, SVR, SVR+, OFO, OFO+, CEN) run. **E is reported whatever the outcome.**

## Scarcity criterion (plant model only; no controller result used)

**Bound.** The REF start point (t = 0), with all events of the candidate applied.
- **TS-a2:** area 2's own TS resources at their voltage-raising limit:
  - the TS-DER at IEEE 6 at Q_max(P) (VDE-AR-N 4120 box, 205 Mvar);
  - the three area-2 MSCs fully in, MSRs out;
  - neighbouring areas and STS DER as at t = 0.
- The resulting area-2 level (area-mean deviation from 1.03 pu) is an upper bound for any
  controller of area 2 without access.
- **+STS:** TS-a2 plus the area-2 STS DER at a common fraction α of Q_max until the level is back
  at 1.03 pu, with coupler taps frozen.

**Criterion.**
- The TS-a2 level is ≤ −10 mpu, i.e. no controller without access can restore the area level.
- The level is restorable with ≤ 25 % of the area-2 STS DER capability at STS voltages
  ≤ 1.10 pu (`dso_v_max_pu`).

**Measured** (`scarcity_E_bounds2.py` and `.out`, shared notes folder of the Case-2 session,
`sts_access_gain_scripts/`). Levels in mpu.

| end state | frozen level | TS-a2 level | STS Q to restore | STS V max there | criterion |
|---|---|---|---|---|---|
| D: G3, 4–14, 250 Mvar @ IEEE 4 | −17.0 | +24.8 | – | – | not met |
| G3, 9–39, 250 @ IEEE 4 | −17.6 | +47.2 | – | – | not met |
| G3, 9–39, 400 @ IEEE 4 | −23.6 | +13.8 | – | – | not met |
| G3, 9–39, 500 @ IEEE 4 | −27.5 | −6.5 | +28 Mvar | 1.063 | not met (> −10) |
| **E: G3, 9–39, 600 @ IEEE 4** | −31.6 | **−27.0** | +115 Mvar (α = 0.13) | 1.070 | **met** |

The deficit size was chosen by this criterion: 600 Mvar is the first step of the series that
meets it.

## Expected results (hypotheses, written before the run)

1. **Area 2, after 212:20:**
   - OFO cannot restore the area level: its area-2 level error is ≥ 15 mpu, a margin below the
     bound because the bound is computed at the t = 0 operating point.
   - OFO+ restores it to ≤ 5 mpu.
   - OFO+ − OFO in area-2 e_v is ≤ −5 mpu.
2. **Aggregate e_v over 6 h:** OFO+ − OFO is at least twice D's (≤ −0.9 mpu). The estimate from the
   bound is −2 to −4 mpu (about −30 to −40 %).
3. **SVR+ − SVR < 0** in area 2 after 212:20. Without G3, SVR has no area-2 AVR lever; SVR+ lifts
   the pilot (IEEE 8) through STS 1–3.
4. **No sign is predicted for OFO+ − CEN.** CEN can also raise the neighbouring areas' machine
   voltages.

If a result contradicts a hypothesis, that is reported as such.

## Limits of the statement

- **600 Mvar is large.** It is about 3× IEEE 4's base reactive load, so E is a stress test of the
  scarcity condition, not a typical event.
- **The bound is optimistic for OFO.** OFO trades the level against the spatial shape, so its
  level shortfall can exceed the bound's.
- **One operating point and one noise seed.** A 400 / 500 Mvar series (OFO, OFO+, CEN) would show
  where the gain sets in. It is not part of this declaration.
