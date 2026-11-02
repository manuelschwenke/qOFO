# Declaration: Case-2 stress scenarios B and D (written before any run)

Date: 2026-10-30, before the first run of record of either scenario under the harmonised
device limits.
Decision: author, 30 Oct (weekend plan, `docs/analysis/2026-10-30_weekend_plan.md` in the
`case2-ref` worktree, §2.4).

## What is declared

Both scenarios run with all six variants (LOC, SVR, SVR+, OFO, OFO+, CEN). **Both are reported
whatever the outcome.** Scenario A stays the reference scenario of ch:case3 (author decision
2026-10-27); B and D are additional evidence on reactive-power scarcity.

| | B (defined 2026-10-27) | D (new) |
|---|---|---|
| 30:40 | G3 out (pandapower gen 1, area 2's only machine) | same |
| 121:40 | line 4–14 out (line 7) | same |
| 212:20 | 250 Mvar at IEEE 8 (load bus 7) | 250 Mvar at **IEEE 4** (load bus 3) |

All events persist to the end of the 360-min horizon. The offsets are those of A, strictly between
two TS dispatches.

## Why these two (rule, not result)

- **Scarcity mechanism.** The outage of area 2's only machine. The distributed slack replaces its
  P, so the power flow stays feasible.
  - The alternative "machine near rating" is not valid in the present plant: the plant's machine
    Q box is static at ±0.5 S_n and does not shrink with P.
- **Load bus of D.** The location rule of `docs/analysis/2026-10-30_sts_access_gain.md` §3.10
  picks IEEE 4. It has the highest STS-to-TS sensitivity ratio among the buses without a TS tap,
  and it has no MSC.
- **B is kept as defined.** It was declared and screened on 2026-10-27, before any access-gain
  analysis.

## Scarcity criterion (plant model only; no controller result used)

- **Criterion:** the open-loop area-2 RMS deviation from 1.03 pu after the load event must be at
  least that of A's area-2 event (32.8 mpu).
- **Set-up:** t = 0 of the reference period, every control frozen.
- **Measured** (`scarcity_openloop.py`, session scratchpad, 2026-10-30):
  - B 48.7 mpu;
  - D 43.0 mpu;
  - A's area-2 event 32.8 mpu.
- Both B and D satisfy the criterion.

## Expected sign (hypothesis)

- **OFO+ − OFO.** e_v(area 2) after the load event is < 0, with a magnitude larger than in A's
  corresponding window (A, weights c08b, 1.07 record: −0.34 mpu after 212 min).
- **SVR+ − SVR.** e_v(area 2) after the load event is < 0.
- **Reason:** without G3, neither law has an AVR lever in area 2. OFO is left with the TS-DER at
  IEEE 6, the MSCs and the taps at IEEE 12; OFO+ adds STS 1–3.
- If the result contradicts the hypothesis, that is reported as such.

## Device set and limits these runs use (code of the weekend branch)

- AVR range [0.95, 1.05] pu for every variant that moves AVR references.
- Minimum time between two taps:
  - TS taps: 15 min in every variant (local relays 1.03 ± 0.01 pu on the HV side in LOC, SVR,
    SVR+);
  - STS couplers: 120 s in every variant.
- Banks:
  - MIQP-dispatched in OFO, OFO+ and CEN;
  - local rule in LOC, SVR and SVR+ (MSC in below 1.01 pu, MSR in above 1.05 pu on the coupler's
    primary side, each after 3 consecutive 20-s steps beyond the threshold);
  - 30-min holding time and MSC/MSR exclusion in every variant.
