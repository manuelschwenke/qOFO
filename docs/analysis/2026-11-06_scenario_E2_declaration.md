# Declaration: Case-2 scenario E2, area-2 scarcity without the 9–39 opening (written before any run)

Date: 2026-11-06, 10:39 (commit 0c80a62), before the first run of E2 (launched 10:39:41).

## Request (author, 2026-11-06)

- "is maybe the 'queue of events' to hard? opening line 9-39 and G3 failure? maybe we should only
  let G3 drop, let it return, then do a load step of 600 Mvar + one line opening, that does not
  have to be 9-39"
- "if not, we can have G3 outage and load step in parallel for some time, but not together with
  line trip which makes 3 events parallel, very hard"
- "then do such a scenario as i described, with 9-39 in operation, and maybe G3 and load step
  overlapping, then G3 coming back, then a line drop later"

## Why

E (G3 out 30:40, 9–39 open 121:40, 600 Mvar at IEEE 4 212:20, all persisting) drives the plant
far from the operating point of the controllers' frozen sensitivities.

- CEN ran a period-2 cycle between the 9–39 opening and the load step (run 0019).
- SVR reached 0.868 pu.
- The plant-only check (shared notes `cen_mismatch.py`, `cand_mismatch.py`) shows where this
  comes from: the 9–39 opening, not G3 or the load. The scarcity comes from G3 and the load:
  - Without 9–39, area 2's own-resources bound stays at −28 mpu.
  - With 9–39 open, CEN's mismatch loop gain rises to 3.1 at its former ceiling λ = 1.405 (2.1 at
    the new 0.90).

## Schedule

| time | event | element | phase |
|---|---|---|---|
| 30:40 | G3 out (with its machine transformer) | gen 1 | 40 s |
| 91:40 | 600 Mvar connected at IEEE 4 (E's location rule and size) | load bus 3 | 100 s |
| 152:20 | G3 back in service | gen 1 restore | 140 s |
| 241:00 | line 3–4 out | line 4 | 60 s |

- At most two disturbances are active at a time.
- G3 and the load step overlap for 61 min (settling p95 of the v4 weights: 23 min).
- The load step persists to the end.
- 9–39 stays in service.
- The events lie between transmission dispatches at distinct phases, on the 20-s step grid
  (test `test_E2_keeps_9_39_and_at_most_two_disturbances_at_a_time`).
- Runner check: the dormant 600-Mvar load exists at bus 3, line 4 is IEEE 3–4, G3 is committed,
  and all four events are registered.
- The rest is as in E: reference period from 08:00, v4 weights, dead band 0.010 pu, machine taps
  held in L2/S1/S2, noise seed 20261027, all six variants.
- CEN at its new ceiling λ = 0.90 (`variants.M_LAMBDA_TARGET`, author "Use 0.90 instead").
- A second CEN at the selected design's λ = 1.405 (`Msel`, author: "would this be the best rule
  anyway, to have a similar lambda than in the OFO scenarios?") runs in its own run alongside.

## Choice of the line (rule declared before the numbers were read for it)

- **Candidates:** area 2's lines and ties other than 9–39 (author), plus the A tie 17–27 as a
  remote control.
- **Rule:** the candidate whose opening leaves the largest area-2 deficit at the TS-a2 bound in
  the end state (G3 back, 600 Mvar at IEEE 4), subject to:
  - every controller's mismatch loop gain < 2 in every state of the schedule (CEN at 0.90; the
    TS-OFO of each area under O2 and O3);
  - a frozen-control area-2 minimum ≥ 0.90 pu.
- **Line 4–5 is excluded as degenerate:** it separates IEEE 4 from its stronger feed, and the
  deficit shows as spatial spread (e 34 mpu) at a small level deficit (−6.6 mpu).

## Plant-only bounds (t = 0 operating point; shared notes `candE.py/.out`, `candE_stage1.out`)

"Level" is the area-2 (IEEE 3–14) mean deviation from 1.03 pu. TS-a2 includes G3's AVR at 1.05 pu
when G3 is in service (`scarcity_F_bounds2`'s TS-a2 left machines at their t = 0 set-points: right
for E, wrong with G3 in).

| stage | frozen level (min V) | TS-a2 | TS-all | STS share to restore |
|---|---|---|---|---|
| 1: G3 out (30:40–91:40) | −4.9 (1.019) | +56.0 | +97.7 | — (not scarce) |
| 2: G3 out + 600 (91:40–152:20) | −58.2 (0.937) | −28.3 | +10.3 | 25 % |
| 3: G3 back + 600 (152:20–241:00) | −41.8 (0.952) | −18.0 | +6.0 | 29 % |
| **4: + 3–4 open (241:00–360)** | −59.2 (0.915) | **−30.5** | **−10.4** | 42 % |

End states of the other candidates (TS-a2 / TS-all / STS share / frozen min V):

| open line | TS-a2 | TS-all | STS share | frozen min V |
|---|---|---|---|---|
| 2–3 | −23.5 | −2.1 | 35 % | 0.939 |
| 4–14 | −22.7 | +1.3 | 37 % | 0.920 |
| 3–18 | −19.9 | +3.0 | 32 % | 0.949 |
| 17–27 | −19.8 | +3.9 | 32 % | 0.949 |
| 14–15 | −19.2 | −1.6 | 26 % | 0.945 |
| 9–39 (excluded) | −14.5 | +12.9 | 19 % | 0.945 |

The STS share exceeds E's 17 % partly because G3, held at 1.05 pu by its AVR, absorbs part of the
STS injection.

## Mismatch loop gain (max Re eig G_w⁻¹ H_mᵀ D_v H_t, continuous columns; shared notes `cand_mismatch.out`)

| state | CEN 0.90 | O2 areas 1/2/3 | O3 areas 1/2/3 |
|---|---|---|---|
| intact | 0.90 | 0.14 / 0.44 / 0.45 | 0.14 / 0.49 / 0.47 |
| G3 out | 1.39 | 0.15 / 0.71 / 0.49 | 0.15 / 0.80 / 0.51 |
| G3 out + 600 | 1.45 | 0.15 / 0.80 / 0.50 | 0.15 / 0.85 / 0.52 |
| G3 back + 600 | 0.93 | 0.14 / 0.47 / 0.46 | 0.14 / 0.51 / 0.47 |
| G3 back + 600 + 3–4 | 0.99 | 0.15 / 0.60 / 0.46 | 0.15 / 0.63 / 0.47 |
| (old E state) | 2.11 | 0.17 / 1.33 / 0.55 | 0.16 / 1.37 / 0.56 |

Msel (λ 1.405) scales CEN's column by 1.56: 2.17 (G3 out) and 2.26 (G3 out + 600) are marginal. Old E
showed no cycle at a static 2.17.

## Expected results (hypotheses, written before the run)

1. **Stage 1** (not scarce): O3 − O2 in area-2 e_v small (|Δ| ≤ 2 mpu).
2. **Stages 2 and 4** (scarce): O3 − O2 < 0 and S2 − S1 < 0 in area-2 e_v. Stage 4 is the largest,
   since TS-all also falls short there: only the STS restore the level.
3. **Stage 3:** still scarce (−18 mpu); the access gain is smaller than in stages 2 and 4.
4. **CEN** (λ 0.90) has the lowest area-2 e_v in stages 2–4 and no period-2 cycle (mismatch gain
   ≤ 1.45).
5. **Msel** (λ 1.405) may cycle while G3 is out (stages 1–2); after G3's return it should not
   (≤ 1.6).
6. **LOC** reaches an area-2 minimum near 0.92 pu after the line trip.

If a result contradicts a hypothesis, that is reported. E2 is reported whatever the outcome.

## Limits

- The bounds are static, at one operating point; the run drifts with the profile.
- One period, one noise seed. The cross-scenario statement remains the MC campaign's.
