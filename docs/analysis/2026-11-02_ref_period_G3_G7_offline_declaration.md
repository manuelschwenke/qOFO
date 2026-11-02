# REF period with G3 and G7 offline (area 2 without a machine, area 3 with one): selection rule

**Status: DECLARED.** The author approved this rule on 2026-11-02 at ~15:00 ("Approve and
commit"). It was committed by Claude Code **before** the candidate search and the screen, as the
GA rule was (`2026-10-31_ref_period_G3_offline_declaration.md`). Names (period "H", scenario
"HA") are placeholders.

## Why

- **Purpose** (author, 2026-10-31): realistic scenarios in which a TS area is short of its own
  reactive reserve and the STS can help through the cascade.
- **GA covers area 2 only.**
  - G3 is offline there, so area 2 has no synchronous machine.
  - Area 3 keeps both coal units: in GA, its machine reserve has a minimum of 368 Mvar under O2,
    after A's G4 trip.
- **Commitment rule in area 3** (`compute_merit_order_dispatch`, merit order then gen index):
  - G4 (Coal, 800 MVA) is committed if $\hat R_3 \ge 320$ MW;
  - G7 (Coal, 700 MVA) is committed if $\hat R_3 - 680 \ge 280$ MW, i.e. $\hat R_3 \ge 960$ MW,
    where $\hat R_3$ is area 3's peak residual over the window.
  - G7 is the marginal coal unit.
- **Year scan** (2026-11-02, session scratchpad `area3_coal_scan.py`, base as in the Case-2 REF
  runs):
  - over the REF candidate windows, G7 is committed in 86 %;
  - G3 off with exactly one area-3 coal unit committed occurs in 12.3 %;
  - G3 and both coal units off occurs in 1.6 %.
- **The alternative, an added renewable in area 3, was considered and not proposed.**
  - Wind (WP10) lowers G7's commitment share only by about 8 pp per +500 MW; PV hardly at all.
  - It would change the network, and with it the commitment in REF A and in the tuning window
    W1 (both 5 Jan), and so the Ch 9 base.
  - A TS-connected park mostly replaces machine Q with converter Q in the same area.
- **The condition exists in the base already.** A period chosen by rule gives it without
  changing the network or the tuning.

## Rule

1. **Candidates.** As for GA:
   - 6-h windows of the profile year (REF horizon 360 min);
   - starting at a full hour between 06:00 and 12:00, in calendar order;
   - outside `run_mc.EXCLUDED_DAYS`.
2. **Commitment** by the base rule over the window. The window qualifies if:
   - G3 is not committed;
   - G7 is not committed;
   - at least three synchronous machines are committed (G1 aside), as in GA.
   - With five machines, this means exactly **{G10, G9, G4}**: area 1 keeps two machines, area 3
     keeps G4 only.
3. **Screen**, as for GA. In calendar order, the first qualifying window that passes both checks
   is selected:
   - **S1:** LOC, no events, the full 6 h at dt = 180 s: every power flow converges.
   - **S2:** the cumulative post-event states of HA (item 5) are solvable under LOC, with the
     plant loops as in the run, at the window's initial operating point (`event_state_check`).
4. **Scarcity check** (plant model only; no controller result used). The set-up is that of the
   B/D criterion (`2026-10-30_scenarios_B_D_declaration.md`):
   - t = 0 of the selected window, every control frozen;
   - open-loop area-3 RMS deviation from 1.03 pu after HA's area-3 load step.
   - Criterion: at least 32.8 mpu, the value of A's area-2 event.
   - If 250 Mvar misses it, the step is enlarged to the smallest of 400 / 600 Mvar that meets it
     **and** passes S2. If none does, HA is reported as "not scarce at this period", and the
     author decides.
5. **Scenario HA** on the selected period: A's schedule, with A's G4 outage replaced by a load
   step in area 3 at the location-rule bus. Offsets are those of A, strictly between two TS
   dispatches. All events persist.

   | time | event | area |
   |---|---|---|
   | 30:40 | connect 250 Mvar (step size per item 4) at **IEEE 24** (load bus 23) | 3 |
   | 121:40 | connect 250 Mvar at IEEE 12 (load bus 11), as in A | 2 |
   | 212:20 | trip line 25 (17–27, tie between areas 1 and 3), as in A | 1/3 |

   - **IEEE 24** comes from the location rule (`case2-ref` worktree,
     `docs/analysis/2026-10-30_sts_access_gain.md` §3.10, index 1.47).
     - It is the area-3 bus where the STS is best placed relative to the TS-DER and MSC.
     - STS 4 couples there (trafo_9), there is no MSC (MSR only), and no TS tap acts.
   - **A's G4 outage is not kept.** In this period it would trip area 3's last machine. That case
     (8 Jan 10:00, G10 + G4 committed, G4 trip) had **no power-flow solution** on 31 Oct, under LOC
     and OFO+ (`2026-10-31_commitment_base_status.md`; diagnosed as plant numerics with the Q(V)
     droops).
6. **Runs.**
   - All six variants, run like A and GA, with the parameter set of record at run time
     (recorded in the manifest).
   - Reported together with A and GA, whatever the outcome.

## Expected sign (hypotheses, stated before any run)

- **O3 − O2:** e_v(area 3) after 30:40 < 0. Without G7, OFO's levers in area 3 are G4's AVR,
  W3, W4, the taps and the banks. OFO+ adds STS 4.
- **STS 4** is recruited under O3. Its interface injection after 30:40 exceeds that of GA at the
  same instant.
- **Area 2** behaves as in GA up to 121:40. Differences come from the reduced support of area 3.
- If a result contradicts a hypothesis, that is reported as such.

## Limits stated in advance

- **One period.** It illustrates the condition "G3 and G7 offline"; it is not a sample of it.
- **Area 3 has one STS.**
  - STS 4's DER provide at most about 189 Mvar upward. Its reported interface injection reached
    up to 205 Mvar at t* in GA.
  - A deficit beyond that is a capacity result of the STS, not a failure of the control scheme.
- **The N-1 outage of area 3's last machine is not tested.** HA avoids a known plant-numerics
  limit; this is a limitation of the scenario, not a property of the controllers.
- **S2 is solved at the initial operating point,** not at the event instants. The run itself may
  still fail later, and a failure of single variants in the run is a result, as in GA.
- **The commitment depends on the Case-2 network base** (DSO_3 ×2 load and DER), as for GA.

## Implementation (after approval)

- A search driver modelled on the GA screen, which applies items 1–4 and writes the screened
  windows and their results to a table in this note.
- Period and scenario entries in `experiments/ch_11_case2_multisystem/ref/scenarios.py`: HA with
  `start` set to the selected window. After that, HA can also be selected in
  `run_multi_system_ofo.py` via `SCENARIO = "HA"`.
