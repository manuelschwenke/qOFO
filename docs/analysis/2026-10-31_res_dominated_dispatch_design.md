# Design note: renewable-dominated operating points through unit commitment (for author decision)

Date: 2026-10-31, ~01:00. Author of the note: Claude Code (side session), for M. Schwenke.
Trigger: author, 31 Oct, ~00:30. "Since the dispatch is our own construction, it is fair and
scientific to change the dispatch assumptions to have more realistic renewable-dominated scenarios,
in REF and in the MC."
Status: **proposal; nothing implemented.**

## 1. Facts: what the present dispatch does

Code of record: `F:\qOFO_weekend`, 8ebc032.

- **Dispatch rule.** `compute_zonal_gen_dispatch` (`network/ieee39/zonal_balancing.py`) spreads each
  zone's residual load (load − DER) over **all** machines of the zone in proportion to S_n:
  - P is capped at 0.85 S_n, with cross-zone spill;
  - **P_min = 0** (`multi_tso_dso.py:2021`; the comment says 20 %, the code multiplies by 0.0).
- **No commitment.** No machine is ever taken offline. A machine at P = 0 keeps its full plant Q
  range of ±0.5 S_n (`build.py:167–168`).
- **Full profile year** (`year_dispatch.py`, hourly, no simulation):

  | | |
  |---|---|
  | RES energy share (DER / load) | 31 % |
  | hours with RES share > 0.5 / > 0.7 / > 0.9 | 24 % / 13 % / 4 % |
  | REF point (5 Jan 08:00) | RES share 0.46; G3 at 16 % of S_n |

  | machine (area) | hours with P/S_n < 0.2 / < 0.4 / < 0.5 | median P/S_n at RES share > 0.7 |
  |---|---|---|
  | G3 nuclear 800 MVA (2) | 25 % / 30 % / 33 % | 0.00 |
  | G4 coal 800 MVA (3) | 6 % / 14 % / 19 % | 0.21 |
  | G7 coal 700 MVA (3) | 6 % / 14 % / 19 % | 0.21 |
  | G9 nuclear 1000 MVA (1) | 85 % / 100 % / 100 % | 0.09 |
  | G10 hydro 1000 MVA (1) | 85 % / 100 % / 100 % | 0.09 |
  | G1 equivalent 10 GVA (1) | 85 % / 100 % / 100 % | 0.09 |

- **Reading.**
  - The profile year already contains renewable-dominated hours.
  - What makes them reactive-power-rich is the dispatch construction: every machine stays
    synchronised at any P, with its full Q range.
  - Area 1's low loading of G9/G10 all year is an artefact of the proportional rule: G1's 10 GVA
    takes most of the zone's residual. A merit order would load the nuclear unit first.

## 2. Proposal

| # | element | proposed default | why (a priori, not from results) |
|---|---|---|---|
| P1 | **Technical minimum** | nuclear 0.5, coal 0.4, hydro 0.1 of S_n | typical minimum stable generation of these unit types; literature anchor to be cited in Ch 7 |
| P2 | **Commitment rule** | at the episode start, a unit whose dispatch would lie below its minimum over the horizon is decommitted. Its residual goes to the zone's committed units, otherwise to the other zones (spill), otherwise to G1. Commitment is fixed for the horizon (3 h MC, 6 h REF), with the machine transformer out as for a trip. | unit commitment works on hourly to daily time scales, with minimum up/down times of hours |
| P3 | **G1** | always committed, Q range unchanged | it is the external-system equivalent, not a unit. Limiting its Q exchange is a separate decision (§4) |
| P4 | **Merit order inside a zone** | nuclear → coal → hydro → G1 (import), instead of the S_n-proportional split | removes the area-1 artefact. **Optional**: without it, area 1 runs on G1 alone |
| P5 | **RES level** | no scaling of DER; the profile year as it is. A second REF period from the top RES decile ("REF-R") | uses the data's own renewable-dominated hours. Scaling DER is a further assumption that would also change line flows |
| P6 | **Where it applies** | **a second, declared regime in Case 2**: REF-R (A-type events) and MC-R (the same MC draw with commitment), plus E under commitment. Weights unchanged (c08b) | no tuning on test data: the controllers meet a regime they were not tuned on. The present results stay as the "all units online" regime |

**Expected effect (hypothesis).**
- In about 25–33 % of MC episodes area 2 has no synchronous machine, and in about 13 % areas 2
  and 3 both run on IBG, banks and STS alone. That is the condition under which access should pay
  off.
- The "+" gain should grow with the RES share and with the deficit. It is reported against the
  episode's plant margin (TS-a2 bound, §9 of the weekend plan).
- The DER Q range follows VDE-AR-N 4120. At low P (calm, night) both TS-IBG and STS DER lose Q, so
  the STS benefit is itself time-dependent. That is realistic and stays.

## 3. Implementation (after approval)

- `zonal_balancing.py`: technical minima per fuel type, commitment pass, optional merit order.
  Default = the present behaviour, so every existing run reproduces bit-exactly.
- Runner: a config field `unit_commitment` (off by default). Decommitted units and their machine
  transformers are out of service from t = 0.
  - **To be checked:** every init path (distributed-slack weights, TS-OFO actuator sets, SVR AVR
    sets, CEN, Ch 9 tools) with machines out of service at t = 0. Today an outage only ever happens
    by a trip during the run.
- `run_ref` / `run_mc`: a regime switch, recorded in the run manifest.
- Smoke tests: every variant 10 min in REF-R; an MC episode with G3 decommitted; bit-exact
  reproduction of one existing REF variant with the switch off.
- Effort: about 4–6 h. Runs from Sat afternoon, when the e5power MC (≈14:00) and STEP (≈17:30)
  end:
  - REF-R plus E under commitment: 12 runs ≈ 5 h on 12 workers;
  - MC-R: ≈ 280 worker-h ≈ 14 h on 20 workers, i.e. Sunday.

## 4. Decisions needed

1. **Scope:**
   - a second regime in Case 2 only, weights unchanged (recommended); or
   - replace the base dispatch everywhere. That means the full Ch 9 procedure plus Cases 1 and 2,
     ≈ 1,700 worker-h; if chosen, before the planned full Ch 9 rerun.
2. **Technical minima** P1, and **commitment** fixed per episode (P2).
3. **Merit order** (P4): yes or no.
4. **G1:** unchanged, or a limited reactive exchange, e.g. Q ∈ ±0.1 S_n, standing for a
   renewable-dominated neighbouring system as well.
5. **RES level:** profile year as is with a top-decile REF-R period (recommended), or DER scaled
   by a declared factor.

## 5. Risks

- **Post-hoc appearance.** The change follows the access-gain discussion. Mitigation:
  - every element has an a-priori anchor (§2);
  - the present regime stays reported;
  - hypotheses are written before any run.
- **Model scope.** A quasi-static model with no inertia, short-circuit power or frequency
  stability. A system with one or no synchronous machine would in reality need grid-forming IBG or
  synchronous condensers. That has to be stated as a limitation.
- **Untested init paths** with machines out of service from t = 0 (§3).
- **Area 1 has no STS.** Its result depends on G1 (decision 4).

## 6. Author decisions (31 Oct, ~00:40) and implementation

**Decisions.**
- **D-R1:** replace the base case **everywhere** (not a second regime).
- **D-R2:** merit order: yes.
- **D-R3:** limit G1.
- **D-R4:** start new REF and MC on 20 workers per host. STEP is left for now.
- **D-R5:** ensure convergence before the runs. Re-tune if necessary (author: "If yes, you can also
  do that").

**Values chosen in the session (a priori, recorded here):**
- Technical minima: nuclear 0.5, coal 0.4, hydro 0.1 S_n. Merit order: hydro → nuclear → coal →
  import.
- G1 Q range ±500 Mvar, the range of one 1000-MVA unit of the system. The renewable-dominated
  neighbour supports the boundary like one large machine, not like an infinite bus.
- Commitment fixed per run window, from the zone's peak residual.

**Implementation:** branch `res/commitment-base-2026-10-31` (cf2f42a and later), daily log
`docs/daily_log/10_2026/2026-10-31_commitment_base.md`.

**Re-tuning.**
- The weights were selected on design windows W1–W4 and W6, whose operating points change under
  commitment. Strictly, the selection procedure must therefore be repeated on the new base.
- **Now:** the declared re-validation of 30 Oct (same rule, §5 of the weekend plan) is repeated on
  the new base and gates the MC. REF runs speculatively on c08b, as on 30 Oct.
- **Later:** the full Ch 9 procedure stays planned. If it selects other weights, REF and MC are
  repeated.
