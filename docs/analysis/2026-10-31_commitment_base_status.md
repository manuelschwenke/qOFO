# New base (unit commitment, limited G1): status of the night 30/31 Oct, and decisions needed

Author: Claude Code (side session), for M. Schwenke. Written 2026-10-31, ~03:30.
Branch `res/commitment-base-2026-10-31`: cf2f42a, c05da69. Daily log
`docs/daily_log/10_2026/2026-10-31_commitment_base.md`.

## 1. Short answer

- **Implemented and committed:**
  - merit-order dispatch with unit commitment (technical minima per fuel type);
  - G1 limited to ±500 Mvar in the plant;
  - four robustness fixes found on the way.
- **The old base is reproduced exactly** when the new flags are off.
- **Not met: "ensure the runs converge" (author's condition).**
  - **In renewable-dominated periods**, the power flow often has no solution after an N-1
    disturbance.
  - **With G1 at ±500 Mvar**, LOC (no coordination) loses voltage stability in heavy periods even
    without any event. This happens in two Ch 9 design windows (W1 evening peak, W6).
- **Therefore not launched:** the new MC and the Ch 9 re-validation.
- **Queued:** only REF A, E, B, D at the reference period. They converge, and at that period all
  five machines are committed. They start when the old-base jobs end.
- **Old-base jobs untouched.** STEP, MC and the queued old-base E keep running: stopping them was
  denied in auto mode.

## 2. Evidence

Results are from `conv_check*.py`, `event_state_check.py`, `ideal_probe*.py` and
`year_commit.py` in the session scratchpad, copied to the shared notes folder. Unless stated
otherwise, LOC at a coarse 180-s step, which never hits the 20-s event times. The event states
were therefore solved separately.

| case | committed machines | result |
|---|---|---|
| REF period 08:00–14:00, no events (A/B/D/E schedule) | all 5 | ok |
| post-event states of A, B, D, E at the REF start (LOC; OFO+ for E) | all 5 | ok (E: V_TS 0.927–1.031 pu after 600 Mvar, G1 Q +381) |
| W1, 5 Jan 08:00, 12 h | all 5 | **fails at ≈17:00** (V_min 0.92) |
| W3, 4 Jan 08:00, 12 h | all 5 | ok |
| W2, 22 Feb 13:00, 12 h; W4, 26 Nov 10:00, 12 h | all 5 | ok (earlier code; G1 limit not binding) |
| W6, 18 Jul 14:00, 12 h (high export) | 4 (G9 off) | **fails after 75 min** (G1 at +500 Mvar, V_min 0.91) |
| 8 Jan 10:00 (G10 + G4), G4 trip | 2 → 1 | **no solution** (LOC and OFO+) |
| 13 Mar 09:00 (G10 only), 250 Mvar at IEEE 12 / 9–39 out | 1 | **no solution**, even with G1 unlimited and the DER loops frozen |
| 13 Mar 09:00, no event | 1 | fails after ≈213 min (control loop does not converge) |

**Diagnosis.**
- **8 Jan, G4 trip** (`ideal_probe`):
  - with the TS-DER as ideal voltage-controlled sources (PV at 1.03 pu, within their
    VDE-AR-N 4120 Q range) and G1 at ±500, the post-trip state **is solvable**
    (V_TS 1.011–1.048 pu);
  - with the plant's Q(V) droops it is not. The plant solves the droops as a loop around
    Newton–Raphson, and the first Newton solve after the trip, with the pre-event DER Q, has no
    solution.
  - So this case is a **plant-numerics** limit, not a physical collapse.
  - A PV warm start was tried. It converged the warm-start stage but not the subsequent droop
    solve, so it was removed again.
- **W1 evening and W6:** a capacity effect of the **G1 limit**. Area 1 has no STS. Without its
  infinite-bus support, LOC cannot hold area 1 at peak load. The ±1000 Mvar and unlimited
  sensitivities are in §5 when finished.
- **13 Mar, one machine:**
  - the 250-Mvar step at IEEE 12 has no solution even with G1 unlimited, so it is probably a
    genuine voltage collapse;
  - the one-machine system is not N-1 secure for A's and E's events.

**Commitment statistics over the profile year (3-h MC windows):**
- 5 machines committed in 59 % of windows, 4 in 17.5 %, 3 in 6.5 %, 2 in 14.7 %, 1 in 2.5 %;
- G3 committed in 75 %.

## 3. Decisions needed

1. **G1 limit.**
   - ±500 Mvar (one 1000-MVA unit) makes LOC lose area 1 at peak load (W1 evening) and in W6.
   - **Recommendation:** choose the value by an operability criterion, declared a priori: "every
     design window W1–W6 and the REF period solvable under LOC without events". The smallest
     round value that meets it. §5 gives the W6 test at 1000 / unlimited.
2. **Security of the commitment.** Pure economic commitment leaves states that are not N-1 secure.
   - (a) **Recommended:** a must-run rule for voltage support. Commit the next machine in merit
     order in an area while the area's N-1 screen (machine trip, largest line, 250-Mvar step) is
     not solvable. This is what TSOs do (must-run for voltage and short-circuit power), and it
     keeps RES dominance where it is secure.
   - (b) Decommitted machines as synchronous condensers. This restores most of the Q richness, so
     it is not recommended for the purpose.
   - (c) Accept "no solution" as an outcome. Not recommended: in the 8-Jan case it is numerics,
     not physics.
3. **Plant numerics for weak-grid states.** The droop-around-Newton plant fails where the physics
   has a solution (8 Jan). Options:
   - a continuation (homotopy) on the disturbance;
   - Q(V) inside the power-flow equations. pandapower has no native element for this; it needs a
     design note.
   - With (2a) this matters less, since the must-run keeps states away from the edge.
4. **Ch 9.**
   - The re-validation on the new base can run only once (1) is fixed: W1 and W6 are design
     windows.
   - The full procedure remains planned.

## 4. Launch state

| what | host | state |
|---|---|---|
| old-base STEP (14 workers), Case 1 (6), queued old-base E (6) | e5wimi | running / queued, untouched |
| old-base MC (20 workers) | e5power | running, untouched |
| new-base REF A, then E (6 workers) | e5wimi | queued, starts when the old-base E ends (≈07:00); chain pid 146552 |
| new-base REF B, D (2 × 6 workers) | e5wimi | queued, starts when the old STEP ends (≈17:30) |
| new-base re-validation and MC | e5power | **not launched**: decisions 1–2 first. The branch is deployed to `F:\qOFO_runs\qOFO_res` with the Ch 9 inputs; `results/res_logs/chain_res.py e5power` starts both after the old MC |

## 5. G1-limit sensitivity on W6 (LOC, no events)

(Filled in when finished.)
