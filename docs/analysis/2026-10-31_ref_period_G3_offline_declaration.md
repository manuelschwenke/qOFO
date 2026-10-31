# REF period with G3 offline (area 2 without a synchronous machine): selection rule

Author decision 2026-10-31 ~08:50 ("Yes, do it"), in answer to the proposal: a declared REF
period, chosen by rule, with G3 offline but three or more machines online, on which A's and E's
disturbances still have a power-flow solution. Declared and committed by Claude Code **before
the candidate search and the screen**. New base as of commit 35b447e (merit-order commitment,
technical minima first, G1 limited to ±1200 Mvar).

## Why

- The 6-h REF period (05 Jan 08:00–14:00) commits all five machines. The new base therefore
  changes the existing REF scenarios A, B, D and E mainly in the P split, not in the Q
  resources.
- The renewable-dominated period of RA/RE (13 Mar 09:00, only G10 committed) has no
  power-flow solution after A's load step or E's tie trip, so it cannot carry a REF scenario.
- In the 3-h windows of the profile year, G3 is offline in 25 % and two or fewer machines are
  online in 17 %. A REF period with G3 offline is the case "area 2 without its own
  synchronous machine", in which the STS reactive capability is expected to matter most
  (hypothesis).

## Rule

1. **Candidates.** 6-h windows of the profile year (REF horizon 360 min), starting at a full
   hour between 06:00 and 12:00, in calendar order, outside `run_mc.EXCLUDED_DAYS` (the
   design-window days, the REF day included). These are the same candidates as for RA/RE.
2. **Commitment** by the base rule over the window (merit order, technical minima, peak
   residual of the 6 h). The window qualifies if:
   - G3 is not committed;
   - at least three synchronous machines are committed (G1 aside).
3. **Screen.** In calendar order, the first qualifying window that passes both checks is
   selected:
   - **S1:** LOC, no events, the full 6 h at dt = 180 s: every power flow converges. This is
     the MC prescreen check.
   - **S2:** the cumulative post-event states of A's and of E's schedules are solvable
     under LOC, with the plant loops as in the run. They are solved at the window's initial
     operating point (`event_state_check`, as for the existing REF scenarios on 31 Oct).
   - An outage of a machine that is not committed changes nothing (E's G3 trip is then
     void).
4. **Scenarios** on the selected period:
   - **GA:** A's events (G4 out, 250 Mvar at bus 12, tie 17–27 out);
   - **GE:** E's events (G3 out, line 9–39 out, 600 Mvar at bus 4).
   - All six variants, run like A and E. Reported together with A and E, whatever the
     outcome.

## Limits stated in advance

- S2 solves the post-event states at the initial operating point, not at the event instants
  (30, 120, 210 min). The run itself may still fail later. A failure of single variants in
  the run is a result, as in the MC.
- One period. It illustrates the condition "G3 offline"; it is not a sample of it.

## Result of the search (31 Oct 09:00–09:58)

- **Qualifying windows** (rules 1–2): 222.
- **Screened** in calendar order (`g3off_screen.py`, shared notes folder), all on the fixed
  dispatch 35b447e:

| window | online (besides G1) | S1 | S2 A | S2 E | result |
|---|---|---|---|---|---|
| 01 Jan 07:00–11:00 (5 windows) | G10, G9, G4 or G7 | **fails**: the runner's bootstrap power flow diverges at initialisation | — | — | fail |
| 06 Jan 06:00 | G10, G4, G7 | ok | ok | 600 Mvar: no solution | fail |
| 06 Jan 07:00 | G10, G4, G7 | ok | ok | 600 Mvar: no solution | fail |
| **08 Jan 06:00** | **G10, G9, G4, G7** | **ok** (V_min 1.003) | **ok** (V_TS ≥ 0.931) | **ok** (V_TS 0.936–1.034, G1 +335 Mvar) | **selected** |

- **Also screened**, in the same parallel batch and recorded only:
  - pass: 08 Jan 07:00 and 09:00, 16 Jan 06:00 and 07:00;
  - fail on E's load step: 08 Jan 08:00, 12 Jan 06:00.
- **Selected period: 2016-01-08 06:00–12:00**, scenarios GA and GE in
  `experiments/ch_11_case2_multisystem/ref/scenarios.py`. E's G3 outage is void there.
- **Observation (not part of the rule):** with only three machines and G3 offline, E's 600-Mvar
  step has no plant solution in most of the windows screened, whenever G9 is also offline.
  Area 2's reactive deficit in E thus lies at the edge of what the system without G3 can
  carry.
