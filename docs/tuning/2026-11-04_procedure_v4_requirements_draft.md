# Controller requirements for the next weight selection (v4) — DRAFT

**Status:** collecting the author's definition (2026-11-04), before any code change and before
the re-tuning. Not frozen. Each item is either an author decision (quoted), the v3 rule kept
until the author changes it, or open.

**Why.** Run 0006 (v3 + amendments 1 and 2) selected 8ad82442269b with GO. Its held-out W5
counts 31 tap reversals, against 0–1 per window in W1–W4 and W6; 22 of them begin within
10 min of a bank switching. A tap hold at bank switching is being evaluated in a parallel
session, and the author wants the requirements fixed before the re-tuning.

## Author decisions so far (2026-11-04 ≈ 10:40)

| Rule | v3 (run 0006) | Next | Author |
|---|---|---|---|
| Hunting tolerance (taps / banks) | 1 / 0 | **0 / 0** | "we go back to zero tolerance now" |
| Settling acceptance | p95 ≤ 36 (4 periods), median residual ≤ 1 Mvar | **p95 ≤ 27 (3 TS periods)**, residual ≤ 1 Mvar | "go to 27 please, so 3 TS periods" |
| Go: e_v,TS | ≤ 1.10 × 0.00827 = 0.00910 (relative to v2) | **≤ 0.01** (absolute) | "lets say 0.01" |
| λ_max(B) ceiling | 1.546 = 1.6 / 1.035 | **1.546** (unchanged) | "1.6 actually, correct?", then, after the explanation below, "Keep 1.546" |

### The λ ceiling (answer to the author's question)

- `tuning/stage1_screen.py`:
  - contraction needs 0 < λ_r(B) < 2;
  - the screen is λ_max ≤ 0.8 · 2 = 1.6 (a declared 20 % reservation);
  - the selection ceiling is 1.6 / (1 + ε_op) = 1.546, with ε_op = 0.035, "the finite-bank
    allowance": λ is computed on a finite bank of operating points, and the allowance
    covers operating points outside it.
- Selecting at 1.6 drops that allowance. Unscreened points could then reach about
  1.6 × 1.035 = 1.66, still below 2 (17 % reservation).
- Effect on the bank price floor (bisection on the screen):

  | Weights | Floor at 1.546 | Floor at 1.6 |
  |---|---|---|
  | 8ad82442269b (price 22.8) | 13.5 | 12.7 |
  | ccc91db1de92 (price 19.7) | 10.9 | 10.4 |

### Re-measurement of ε_op (author: "can you remeasure it?"); rule declared before any result

ε_op = 0.035 was measured in August 2026: the worst of six design-to-confirmation shifts of
the worst-window λ, on 12-window banks of the August network. The current screen bank has
6 operating points (W1–W6 at their start times, commitment base). Measurement:

- **Fresh bank F:** 96 conditions of 2016, 12 months × days {8, 22} × hours {02, 08, 14,
  20}. Same capture as the design bank: window-start state, 12-h unit commitment, frozen
  network (cache key `netf8c419bf`).
  - Checked: a fresh capture of W1 reproduces the stored W1 λ to 10 digits.
  - One capture takes 29 s.
- **Design bank D:** W1–W6 of run 0001.
- **Candidates C:** every candidate evaluated in run 0006 (II-a, II-b, II-c, completion)
  and the Stage-I point.
- **Statistic:** per candidate, s(w) = λ_max(D ∪ F) / λ_max(D) − 1.
- **Rule:** ε_op,new = the maximum of s(w) over the candidates near the ceiling,
  λ_max(D) ≥ 0.9 × 1.546 = 1.391. This is the worst value, as in August.
- **Reported, not decisive:**
  - the median and p95 of s;
  - the conditions and layers that set λ_max(D ∪ F);
  - August-style resampling: the max over random 6-point banks from D ∪ F against the
    max over D ∪ F.
- **Consequence (author decides):**
  - ε_op,new ≤ 0.035: 1.546 stays.
  - Larger: either select at 1.6 / (1 + ε_op,new), or screen on a larger bank (D ∪ F)
    with ε re-estimated by resampling.

### Result of the re-measurement (run 0007, 2026-11-04 11:31)

- **Declared rule: ε_op,new = 2.88.** It is not usable as an allowance (1.6 / 3.88 = 0.41).
  - The near-ceiling filter selected all 264 candidates, because the STS layer is at 1.4
    for every candidate.
- **Cause (post-hoc analysis):** the design bank covers only commitment states with the
  zone-2 machine G3 committed.
  - In 10 of 93 fresh points G3 is off. There the TS λ of the candidates has median 3.53
    and max 5.93 (8ad82442269b: 4.16), above the contraction bound 2.
  - With G3 on, a zone-3 machine off still gives shifts of up to about 100 %
    (8ad82442269b: 1.85).
- **Consequence:**
  - ε_op is the wrong instrument for this.
  - Open for the author: should the screen bank include the commitment states (G3 off,
    the zone-3 machine off)? Or should the weights depend on the commitment state (an
    architectural change)?
- Details: daily log `2026-11-01b_…` (entry 11:31) and
  `results/ch_09_parameter_selection/0007_…/analysis_commitment_split.json`.

### Author decision: B1, one robust weight set (2026-11-04 ≈ 11:45)

- Author: "B2 seems to be too complicated … I would prefer B1."
- B1: the screen bank includes the commitment states, at least W1–W6 and the 93 fresh points
  of run 0007, G3-off included.
- **Check (screen only): B1 bounds the bank price from below.**
  - ≥ 29 absolute; 34–40 inside the v3 box.
  - In run 0006 no candidate above 25 committed a bank after an event.
- So B1 with banks in the λ check is incompatible with the bank-engagement requirement.
  Open for the author:
  - **B1a:** banks stay in the λ check; bank engagement becomes a reported metric, not a
    filter or go criterion.
  - **B1b:** the λ check covers the continuous actuators only (precedent:
    `central_lambda_rule = "continuous"`), and a G3-off window joins the search windows,
    so that the zero-tolerance hunting filter judges the banks there.
    - 8ad82442269b would need g_w_der and g_w_pcc ×1.45; its bank price could stay at
      22.8.
- **Author decision (≈ 11:50): "B1b, banks engage".**
  - The λ screen and selection ceiling (1.546) apply over the continuous actuators only.
  - The screen bank includes the commitment states.
  - A G3-off window joins the search windows. Window and event are open (below).
  - Bank engagement stays a filter and go criterion.
- **W7 = 8 Oct 20:00 FAILED the operability check (12:37):**
  - S2 passed: the state after the ramp is solvable, V_TS 1.009–1.041.
  - S1 failed: LOC, no events, 12 h at dt 180 s ends in LoadflowNotConverged. S1e (with
    the ramp) fails the same way.
- **Author proposal (≈ 12:45), under discussion:** take "G3 off" out of the tuning campaign
  and into an additional analysis. Statement: with less synchronous generation the
  sensitivities rise, and in marginal scenarios the controllers need re-tuning. Author's
  question: does G3 then have to be must-run in the merit order?
  - Screen check on the G3-on points of run 0007 (89 of 99):
    - only one has a zone-3 machine off (F1022_14);
    - λ over the continuous actuators (B1b): 220 of 264 run-0006 candidates pass;
      8ad82442269b passes with 1.532;
    - with the banks in λ: 178 pass; 8ad82442269b fails with 1.851 (the zone-3 bank
      column at F1022_14).
  - REF scenarios: A, B, C, D, E (5 Jan 08:00) have G3 on. RA and RE (13 Mar 09:00, only
    G10) and GA and GE (8 Jan 06:00) have G3 off.
  - Assessment (Claude):
    - Valid as a declared design envelope, with the excluded states shown rather than
      hidden. TSOs schedule must-run units for voltage support and system strength, so
      an envelope with one synchronous machine per zone is operationally realistic.
    - A must-run G3 would change the plant dispatch in about 20 % of the hours and would
      remove the purpose of RA, RE, GA and GE.
    - Recommended instead, without a code change:
      - scope restriction: the main campaign is REF A–E, with MC samples restricted to
        G3-on windows (a declared filter);
      - an additional analysis on RA, RE, GA and GE ("low synchronous generation");
      - B1b screen on W1–W6 plus the 89 G3-on points;
      - search windows as in v3 (W1, W3, W4, W6); W7 is not needed.

- **Author decision (≈ 12:50): "Yes, scope filter".** This supersedes W7 and "Replace W1".
  - **Design envelope:** commitment states with G3 committed (one synchronous machine in
    zone 2). No must-run; the merit order is unchanged.
  - **Tuning:**
    - search windows W1, W3, W4, W6 (as v3); completion adds W2; W5 held out;
    - screen bank W1–W6 plus the 89 G3-on points of run 0007;
    - λ ≤ 1.546 over the continuous actuators (B1b).
  - **Case 2:**
    - main campaign REF A–E; the MC samples only windows with G3 committed (a declared
      filter, to be implemented in `run_mc` before the MC);
    - additional analysis "low synchronous generation" on RA, RE, GA and GE with the
      selected weights.
  - **Thesis statement:** with less synchronous generation the sensitivities rise (here
    about ×3 in zone 2). The weights then violate the contraction condition, and the
    controllers must be re-tuned; commitment-dependent weights are future work.
- *(Superseded)* Window sets (author ≈ 12:42: "Replace W1"):
  - Search: W3, W4, W6, W7.
  - Completion: W1 and W2 are added, so the final ranking, the filter and the go criteria
    use W1–W4, W6 and W7.
  - W5 stays held out.
  - Basis: run 0006, where W1 decided no rejection on its own and gave the fewest bank
    credits (W3 has the same event type, one day apart).
- **Replacement rule (declared 12:38, before any result):**
  - Candidates: the G3-off conditions of run 0007, except 8 Jan (GA/GE).
  - Order: night starts (20:00, 02:00) in calendar order first, then day starts in
    calendar order.
  - W7 is the first candidate that passes S2, S1 and S1e with W4's ramp, as for GA.
- *(Superseded proposal, kept for the record)* W7 accepted by the author (≈ 11:55: "Yes,
  use it"):
  - Start 8 Oct 2016 20:00 (F1008_20), 12 h. It differs from the usual commitment only by
    G3 off, and it is a night window. It is not on 8 Jan, the day of the Case-2 scenarios
    GA and GE ("no tuning on the test data").
  - Event: the load ramp of W4.
  - Before freezing, an operability check: start state solvable (the capture converged)
    and a 12-h run under LOC without and with the event, following the GA declaration of
    2026-10-31.

### Hunting dwell (author ≈ 12:53)

- Author: slow reversals (dwell 1–2 h, following the load profile) "does not count as hunting".
- v4 rule: a reversal counts as hunting if its dwell is **< 1 h** (v3: < 2 h) and no new event lies strictly between its two moves. Everything else in the definition stays: events merged within 30 min, the 900 s start-up exemption, and a move at an event reversed minutes later counts.
- The limit cycles of 2026-09-11 (period ≈ 21 min) stay inside the definition.
- Tolerance 0 for taps and banks; the early stop at the first counted reversal.

### Launch (author ≈ 12:53)

- Wait for the parallel session "Tap hunting vs. shunt control" (the tap hold at bank switching), then start the campaign as soon as possible.
- Workers: **30 on e5power, 10 on e5wimi**.

## Kept from v3 until the author changes them

- Hunting definition:
  - reversal with dwell < 2 h and no new event strictly between its moves;
  - events merged within 30 min;
  - a move at an event reversed minutes later counts;
  - start-up exemption 900 s (amendment 2 a).
- Ranking: e_v,TS with the 2 % band, tie-break e_v,DS.
- Engagement:
  - couplers move in every search window;
  - banks commit within 2 h after an event in ≥ 2 search windows;
  - no reserve term.
- Limits:
  - θ_STS,min band [0.005, 0.015] pu, θ_TS,min ≤ 0.009 pu;
  - TSO taps ≤ 15 per 12 h; banks ≤ 12 per day;
  - AVR g_w_gen ≥ about 5·10⁵.
- Windows: search W1, W3, W4, W6; completion on W2; W5 held out.
- Stage II-c, the early stop (now at the first counted reversal), II-a 512 draws,
  II-b ≤ 4 rounds.

## Open (to be defined by the author)

- **The tap hold at bank switching:** duration; which OLTCs (STS couplers, TS taps, both);
  trigger (commit, release, both); scope (the STS below the switching zone, or all).
- Whether a night window joins the search windows. W5 is the only night window and the
  only one that exposed the coupling.
- Whether slow reversals (dwell 1–2 h) that follow the load profile count as hunting:
  9 of W5's 31 are not at a bank switching, 5 of them with dwell 1–2 h.

## Evidence from run 0006 under the decisions so far

- Re-scored at zero tolerance on its stored search-window records:
  - 11 of 264 candidates are feasible;
  - only ccc91db1de92 has banks after events (2 windows): e_v,TS 0.00877 on the search
    windows, 0.00807 on all five, 0 counted reversals on all five;
  - its settling and W5 are not tested.
- The re-score is exact for the evaluated candidates. The early stop was at 2 reversals,
  so every candidate with 0 has complete records. It says nothing about candidates a
  zero-tolerance search would have visited instead.
- 8ad82442269b: settling p95 23.0 meets 27; e_v,TS 0.00799 meets 0.01; its 1 counted
  reversal (W4) fails zero tolerance.
