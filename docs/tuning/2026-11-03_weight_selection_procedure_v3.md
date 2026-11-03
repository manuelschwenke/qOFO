# Controller weight selection, procedure v3 (Ch 8, Experiment B)

**Status:** FROZEN 2026-11-03, before the first simulation of its campaign. It is applied
once, by `experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py`. Its output is the
weight set of the commitment base, whatever it is.

**Why v3.** Procedure v2 (`2026-11-02_weight_selection_procedure_v2.md`) selected
1b945ad40b42 (run 0004). In normal operation that set leaves actuators idle:
- STS couplers make 0 moves in W1–W4 and 4 in W6;
- MSC/MSR banks commit in no window;
- AVRs move 0.003–0.009 pu per 12 h.

The author judged this not the intended behaviour, so the design rules are corrected.

**This procedure is declared after the v2 results** and is a reaction to them. The evidence
is in `docs/daily_log/11_2026/2026-11-01b_ch9_settling_failure_diagnosis.md` (entries of
2026-11-03):
- the winner diagnostics;
- the variants sts010 and shunt27;
- the replays of run 0004.

## Author decisions (2026-11-03)

- **Engagement:**
  - STS couplers must act;
  - banks must commit after events such as load steps and outages;
  - no reserve term in the objective for the standard case study.
- **AVR:** g_w_gen not below about 5·10⁵.
- **Hunting:** hunting tolerance 1 (bank hunting 0).
- **Search scope:**
  - search on W1, W3, W4 and W6, all windows later;
  - II-a 512 draws, II-b up to 4 rounds.
- **Hosts:** 36 workers on e5power and 30 on e5wimi.
- **Go criteria** for the follow-ups below.

## Amendment 1 (2026-11-03 17:30, before the Stage-I verification result)

Declared while run 0005's first verification was running. The diagnostic sts010
(θ 0.01 at the v2 winner's TS weights) had reached p95 25.1, against the verification
margin of 25.

- **Settling budget: 4 supervisory periods** (author: "we could use 4 periods instead of
  3"). Acceptance is p95 ≤ 4 N_inner = 36 and median residual ≤ 1 Mvar; the verification
  margin is p95 ≤ 34 (v2's 2 iterations) and residual ≤ 0.8 Mvar.
- **Coupler engagement ladder** (author: "we could do that"). θ_STS = 0.01, then 0.015,
  then 0.02 pu. The first rung whose Stage-I point passes the verification is the
  Stage-I point; the θ_STS check band is that target ± 0.005 pu.
- Everything else is unchanged.
- **Intent of the 4-period budget (author, 17:31):** "only that we do not filter too hard".
  The report states, besides the 4-period check, whether the selected set also meets the
  3-period limit of v2 (p95 ≤ 27).

## Configuration

As v2, and additionally `shunt_ff_ledger = True`: the bank-release feedforward fix,
commit a8d5e23, a cherry-pick of ac69918.

## Stage I

- **STS weights.**
  - The v2 design, π_q = 10 (v2's verified choice), π_v,DS = 1, λ_STS target 1.4.
  - **Δv_eng,STS = 0.01 pu** (v2: 0.03): the strongest coupler commits at 1.04 pu under a
    uniform deviation from 1.03 pu.
  - The target follows from the winner's per-bus RMS DS deviation, 0.004–0.011 pu per
    window: at 0.03 the commit condition is not met in normal operation.
- **TS weights.**
  - g_v, g_w_der, g_w_pcc and g_w_tso_oltc as v2's Stage I (`stage1_v2_uc_2026_11_pq10`).
  - **Bank price from Δv_eng,shunt = 0.005 pu** (v2: 0.010), giving g_w_tso_shunt ≈ 13.5.
    v2 candidates whose banks committed after events had prices of about 9–21.
  - **g_w_gen = 5·10⁶** (v2: 10⁸), so the II-a box spans 5·10⁵–5·10⁷. In v2, e_v,TS improved
    towards the lower edge of the box, 10⁷.
  - Re-running the TS contraction design with the cheaper bank would raise g_w_der and
    g_w_pcc about fivefold, into a region where no v2 candidate was feasible (0 of 40 with
    g_w_der ≥ 0.3). They therefore stay v2's, and the screen bound applies:
    λ_max(B) = 1.447 ≤ 1.546.
- **Verification:**
  - the settling test of v2 on this single point;
  - acceptance p95 ≤ 25 and median residual ≤ 0.8 Mvar;
  - otherwise the campaign stops.

## Stage II-a and II-b

- **Boxes, seeds and screen as v2:**
  - 512 Sobol draws;
  - spans ±1 decade (g_w_der, g_w_pcc, g_w_gen) and ±0.5 decade (g_w_tso_oltc,
    g_w_tso_shunt);
  - screen λ_max(B) ≤ 1.546 and θ_TS,min ≤ 0.009 pu;
  - II-b rounds of 64 draws, f = 0.5, recentring and halving as v2, at most 4 rounds.
- **Search windows:** W1, W3, W4 and W6, i.e. both generator outages, the load build-up and
  the load shedding.
- **Early stop.** All windows of a candidate stop as soon as its windows together count
  2 unexplained tap reversals or 1 bank reversal.
  - The count uses the filter's own definitions, on the records so far.
  - The Stage-I point is never stopped.
  - A stopped candidate is rejected by the filter regardless.

## Filter and ranking

1. **Limits:** v2's limits at hunting tolerance 1, bank hunting 0; θ_STS,min within
   [0.005, 0.015] pu (0.01 by construction); θ_TS,min ≤ 0.009 pu.
2. **Engagement**, measured on the search windows:
   - (a) STS couplers move in every search window;
   - (b) banks commit within 2 h after an event in at least two search windows.
   - The strictest level that leaves a ranked candidate applies: (a)+(b), then (a), then
     none. The level is reported. The Stage-I point is the reference record and is never
     ranked against a level it fails.
3. **Rank:** e_v(TS) with the 2 % band, tie-break e_v(DS), as v2.
4. **Completion.** After II-b, the final band and the Stage-I point are simulated in W2 and
   ranked on all five windows (W1–W4, W6) with the same filter.
5. **Settling confirmation** in rank order, as v2: p95 ≤ 27 and median residual ≤ 1 Mvar.
6. **Result:** W5 for the report, then `configs/paramsets/ch9_selected_uc_2026_11_v3.json`.
   SELECTED_PARAMSET is not switched; the author decides.

## Go criteria for the follow-ups

Written by the driver to `GO.json` or `NOGO.json`. Only a GO starts the follow-ups:
- regeneration of the settling figure;
- D1 + D2;
- the Ch 10 REF runs.

The criteria:
- e_v(TS) ≤ 1.10 × 0.00827 (v2), on all five windows;
- STS couplers act in every search window;
- banks commit after the events in at least two search windows;
- all limits met.

## Replays that support the reduced search (run 0004, hunting tolerance 1)

- **Search windows.** W1+W3+W4+W6 reproduce the all-window feasible set (116 of 116, none
  wrong) and the all-window leader: subset rank 3, unchanged after completion.
  - Compute ≈ 53 % of an all-window search with the early stop, ≈ 55 % including the
    completion on W2.
- **Early stop at the second reversal:** saves 39 % per candidate on all windows.

## Risks stated before the campaign

- **Banks.** In v2's code, only 1 of the 54 candidates with event-responsive banks was
  feasible at tolerance 1.
  - The ledger fix may change that; its separate verdict on run 0004 was not in when this
    was frozen.
  - The engagement levels keep the campaign from ending without a result, and the go
    criteria report whether banks engaged.
- **Couplers in the settling test.** The pq10 verification steps involved no coupler
  moves; at θ 0.01 they may, and the settling test decides.
