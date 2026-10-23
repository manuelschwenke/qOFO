# Protocol: Experiment B (weight selection) with the banks in the TS-MIQP

**Status:** FROZEN 2026-10-22 22:30 for an overnight run under the author's
delegation ("run the full tuning suite over night", 2026-10-22). Decisions
marked **(C)** were set by Claude under that delegation and are for the author
to confirm; **(A)** are the author's. Execution:
`results/THESIS_ch8_expB_banks_20261022/` (driver `driver.py`, log
`driver.log`). After freezing, deviations go to §10 with date and reason;
§1–§9 are not edited.

Predecessor: `2026-10-16_protocol_experiment_B_rerun.md` (run 2,
Amendments 1–3), whose rules are carried over unless stated here.

## 1. Why a further rerun

| Fact | Evidence |
|---|---|
| The banks are integer inputs of the TS-MIQP since 2026-10-21/22 (`2e06a1b`), without the one-step cap and the hard voltage rows, with MSC/MSR exclusion per STS (`ac8b52f`). The October selection ran with an integrating dispatcher outside the MIQP (`shunt_int_g_w`). | daily logs 2026-10-22 |
| The bank step price `g_w_tso_shunt` therefore acts like a tap price and has to be designed and searched with the other weights. | author decision (a), 2026-10-22 |
| `ch9_selected_10_2026` has silently run at `g_w_tso_shunt = 10` since the fold; with the banks counted its worst λ_max(B) is 2.18 (TS_2, W5), above the OFO bound 2. | `results/2026-10-22_ch8_shunt_engage_table/box_shift.log` |

## 2. Configuration frozen before the run

As §2 of the predecessor (network `rural_700`, DSO_3 ×2; MSR on coupler 1,
MSC on coupler 2, 2 × 25 Mvar each; tap lock 5; V-rescaling on; T_TS 180 s,
T_STS 20 s, dt 20 s; measurement noise as `make_config()`; W1–W6 evented,
12 h, W5 run but not ranked), plus:

| Item | Value |
|---|---|
| Bank law | in-MIQP (`shunt_dispatch='miqp'`), unrestricted, `shunt_sts_mutual_exclusion = True`, holding time 1800 s (not searched) |
| Code state | commit recorded in `code_state/` at launch; the one uncommitted file (`experiments/run_multi_system_ofo.py`, the author's interactive horizon/contingency edit) is captured as a patch. Stage I and Stage II overwrite both fields it changes (`n_total_s`, `contingencies`), Experiment C sets its own horizon. |
| Workers | 30 (other users load ~30 % of the 40 threads) |

## 3. Rules

### 3.1 Stage I
As before (gauge g_v = 4·10⁴, π = 1, τ = 1, λ target 1.0 per layer, taps
from the corrected engagement rule at **D7** 0.0075 / 0.015 pu), plus:
- **Bank price** from the same rule, g_w,sh = max_b 2 g_V |Σ_j H_V[j,b]| Δv_eng,sh,
  **Δv_eng,sh = 0.010 pu (C)**.
- **λ_max(B) keeps one bank column per STS, the stronger (A).** With the
  exclusion only one bank of an STS can move in a given direction.

Result (`stage1/stage1.log`, paramset `stage1_start_10_2026_banks`):
`g_w_tso_shunt` 27.2121, `g_w_der` = `g_w_pcc` 0.09311 (κ_TS +0.25 decade
over the October start), all other weights as in October; worst λ_max(B)
1.000 (TS_2, W5); TS integer floor 0.50 (set by the taps, not the banks).

Reason for 0.010 (C): the smallest candidate at which the bank columns no
longer raise the TS integer floor above the taps' (0.0075 → floor 0.59,
κ +0.38 decade); banks engage after the TS taps (0.0075) and at ≥ 4× their own
per-level zone-mean voltage effect (0.0020–0.0028 pu), so a single step does
not overshoot; the price lies in the event mode (25–34) of the gradient
distribution recorded 2026-10-20 (other weights, older law; indicative).
Table: `results/2026-10-22_ch8_shunt_engage_table/`.

### 3.2 Stage II
Sobol n = 512, seed 20260910 (**N4 (C)**: unchanged; the 9-D design projects
onto the 8 old coordinates with the same low discrepancy), box centred on the
Stage-I point with `DEFAULT_SPANS` (A, Amendment 1 of the predecessor) plus
`g_w_tso_shunt` ±0.5 decade (A). Screen λ_max(B) ≤ 1.5459 with the per-STS
reduction: **119 / 512 survivors** (dry run 22:28). All survivors plus the
starting weights simulated 12 h in W1–W6 (Amendment 2), per-bus metric (D1).

### 3.3 Selection
Filter as before (taps ≤ 30/day per transformer; tap hunting 0; AVR step
≤ 0.005 pu; corridor violations 0; PF failures 0; λ_max(B) ≤ 1.6;
e_q ≤ 3 × e_q(start)), plus bank wear (A: bounded like taps):
- **bank commits ≤ 12 per bank and day (N3, C, provisional)**; the selection
  is also reported at 4, 8, 24 and 48/day. The 12 h records hold every
  commit, so a different budget is a re-selection in minutes, not a rerun
  (then only Experiment C may need new runs);
- bank hunting 0 (opposite commits on one bank < 2 h apart with no new event).
- **W5 (N5, C):** unchanged — the filter uses the ranking aggregate without
  W5, for banks as for taps; W5 is reported separately.

Rank on e_v(TS), 2 % band on e_v(DS). Experiment C (unchanged criterion:
p95 n_settle ≤ 27, median residual ≤ 1 Mvar, 120 steps, `--max-step-mvar 10`)
on the ranking winner; failed candidates are removed and the ranking
recomputed among the remaining feasible ones with the band re-anchored
(Amendment 3). Four candidates are checked in parallel.

## 4. Reported
Selected weights, λ_max(B), feasibility counts and reasons, budget
sensitivity, Experiment C results, bank commits per window, box-edge offsets.
**Orientation (declared, not part of the selection):** `ch9_selected_10_2026`
with the Stage-I bank price (27.21) on the same bank — its `g_w_der` lies
1.17 decades below the new centre, outside the ±1 decade span, and it passes
the screen at that price (1.467).

## 5. Acceptance
The winner is written as **`ch9_selected_10_2026_banks` (N6, C)**.
`SELECTED_PARAMSET` is **not** switched: the author confirms the (C) decisions
first. If no candidate passes, the protocol stops and reports it.

## 9. Decision record

| # | Decision | Value | By |
|---|---|---|---|
| a | How the bank price is set | Stage-I rule + searched ±0.5 decade | A |
| — | Bank columns in λ_max(B) | one per STS, the stronger | A |
| N2 | Δv_eng,sh | 0.010 pu | C |
| N3 | Bank budget | 12 / bank / day, sensitivity 4–48 | C |
| N4 | Sobol n, seed | 512, 20260910 | C |
| N5 | W5 in the wear filter | as taps (excluded with the ranking) | C |
| N6 | Result name | `ch9_selected_10_2026_banks` | C |
| N7 | `g_w_tso_shunt` required | yes | A |

## 10. Amendments after freezing
(none)

## 11. Execution record
- 2026-10-22 ~22:20 — Operating points rebuilt with bank columns and STS
  groups, key `net82afed5b`.
- 2026-10-22 ~22:26 — Stage I, `stage1_start_10_2026_banks` written.
- 2026-10-22 22:28 — Stage II dry run: 119 / 512 survivors (23.2 %).
- 2026-10-22 22:30 — Code state `code_state/` (HEAD a3ce523 on
  `minimal/thesis-core` + verified patch of `experiments/run_multi_system_ofo.py`).
  Driver launched detached (pid 128912): 120 candidates × 6 windows = 720
  units, 30 workers.
- 2026-10-22 ~22:40 — Smoke of the per-window path (start weights, W4, 1 h,
  outside the campaign): row complete incl. bank-wear fields, 302 s wall per
  simulated hour under the current load, i.e. ~20–25 h for the bank.
- 2026-10-23 15:22 — Stage II finished: 720 / 720 units ok, 0 failed,
  120 candidate records, 16.85 h (median 37.4 min per 12 h unit, as in
  October).
- 2026-10-23 15:24 — Selection (`selection/selection_frozen.json`):
  **41 / 120 feasible**. Rejection reasons (a candidate may have several):
  tap hunting 54, interface-Q > 3 × start 25, bank hunting 22, corridor
  violations 12, tap budget 4, bank budget 3. Budget sensitivity
  (`budget_sensitivity.json`): 38 feasible at 4 / bank / day, 41 at 8, 12, 24
  and 48; the top five identical at every budget — the selection does not
  depend on N3. Ranking: **1. faf088d2570e** e_v(TS) 0.00507, e_v(DS)
  0.00844, e_q 115 393, λ_max(B) 1.487, alone in the 2 % band (rank 2
  7295750e02f4 at 0.00532); 3. ed3e7ff339d4 0.00544; 4. starting weights
  3cb66225e1a7 0.00547 / 0.00749 / 1 039 596. No candidate dominates the
  starting weights on both voltage criteria.
- 2026-10-23 15:24–18:44 — Experiment C on ranks 1–4 in parallel
  (`expC/criterion_results.json`): **faf088d2570e PASS** (p95 21.05, per DSO
  17.0 / 14.0 / 23.55 / 11.65, 2 / 120 steps > 27, max 33, median residual
  0.30 Mvar); 7295750e02f4 FAIL (35.0, 3.14 Mvar); ed3e7ff339d4 FAIL (46.0,
  7.01 Mvar, 21 censored); starting weights FAIL (31.1, 2.26 Mvar — as in
  October). The ranking winner passes, so the repeated selection is not
  needed.
- **2026-10-23 18:44 — RESULT: `configs/paramsets/ch9_selected_10_2026_banks.json`
  = faf088d2570e**, not adopted as `SELECTED_PARAMSET` (§5). Weights:
  g_q 0.3561, dso_g_v 4240, g_w_der 0.02355, g_w_pcc 0.3433, g_w_gen
  1.213·10⁷, g_w_tso_oltc 15.67, **g_w_tso_shunt 46.32** (Δv_eng,sh ≈ 0.017 pu
  at the strongest bank), g_w_dso_der 1.144, g_w_dso_oltc 16.22;
  g_w_pcc / g_w_der 14.6. Driver total 20.2 h.
- Box-edge check (§4, offset in fractions of the span from the Stage-I
  point): g_w_gen −0.92 and **g_w_tso_shunt +0.92** above 90 %, to be stated;
  g_q +0.63, g_w_der −0.60, g_w_pcc +0.57, g_w_dso_der +0.36, g_w_tso_oltc
  −0.27, dso_g_v −0.25, g_w_dso_oltc −0.02.
- Bank activity of the winner: no commit in W1–W4, W6; 11 in W5 (unranked).
  Starting weights: 2 / 0 / 6 / 3 / 12 / 0 in W1–W6.
- Orientation (§4, declared, not part of the selection; `orientation/`):
  `ch9_selected_10_2026` weights at the Stage-I bank price 27.21 (key
  cffa2ca108c5): feasible, e_v(TS) **0.00504**, e_v(DS) **0.00697**, e_q
  126 338, λ_max(B) 1.467, bank commits 0 / 0 / 4 / 3 / 20 / 0. It
  **dominates the winner** on both voltage criteria (−0.6 % TS, inside the
  2 % band; −17 % DS) and lies outside the box (g_w_der −1.17 of the span) —
  the pattern of October run 1. Not used; for the author's decision.
