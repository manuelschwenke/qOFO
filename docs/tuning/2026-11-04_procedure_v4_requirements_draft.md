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
