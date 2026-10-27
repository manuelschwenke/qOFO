# Controller weight selection: the procedure (Ch 8, Experiment B)

**Status:** FROZEN 2026-10-25 before its application. Applied once; its
output is the weight set used in Part III, whatever it is. Execution:
`results/ch_09_parameter_selection/0001_2026-10-25_215433_weight_selection_driver_band/` (`driver.py`,
`driver.log`; the run of record, adopted 2026-10-27 from
`results/THESIS_ch8_weight_selection_band_20261025/`).

Development history (not part of the procedure): the procedure was developed
on preliminary campaigns, recorded in `2026-10-16_protocol_experiment_B_rerun.md`,
`2026-10-22_protocol_experiment_B_banks.md` and
`2026-10-24_weight_selection_procedure.md`. The last was applied on the
configuration without a DSO operating band; it is superseded by this one.

## Configuration

Network `rural_700` with DSO_3 ×2 (DER and active load); MSR on the first,
MSC on the second coupling-transformer tertiary, 2 × 25 Mvar each, dispatched
as integer inputs of the TS-MIQP with MSC/MSR exclusion per STS and a holding
time of 1800 s; tap lock 5 iterations; bank V-rescaling on; T_TS 180 s,
T_STS 20 s, dt 20 s; measurement noise as `make_config()` (seed 20260717).
**Every DSO controller holds its observed buses in an operating band of
0.92–1.08 pu**, a margin of 0.02 pu to the grid-code limits, enforced as a
soft output constraint priced by the voltage slack weight (10⁹); the TS
controllers keep 0.90–1.10 pu.
Bank of windows W1–W6 (`tuning.scenarios_mc_v2.CONDITIONS`), one disturbance
each, 12 h. W5 has no DER reactive capability at either level; it enters
neither the filter nor the ranking and is simulated only for the selected
weights, for the report.

## Stage I — analytic starting point

Gauge g_v = 4·10⁴ (TS voltage priority 1 at σ = 0.005 pu); priorities
π_q = π_v,DS = 1, τ = 1; each layer's continuous step prices scaled so that
the worst area and condition has λ_max(B) = 1, with one bank column per STS
in B (MSC/MSR exclusion); integer step prices from the commit rule
g_w = 2 g_V |Σ_j H_V[j,i]| Δv_eng with Δv_eng = 0.0075 pu (TS taps),
0.015 pu (STS taps), 0.010 pu (MSC/MSR banks).
Result: `configs/paramsets/stage1_start_10_2026_banks.json` (the operating
band does not enter H or B, so the point is the same as without it).

## Stage II-a — global screening

512 scrambled Sobol draws, seed 20260910, in a log box centred on the
Stage-I point, spans ±1.5 decades (g_q, dso_g_v), ±1.0 (g_w_der, g_w_pcc,
g_w_dso_der, g_w_gen), ±0.5 (g_w_tso_oltc, g_w_dso_oltc, g_w_tso_shunt);
g_v fixed (gauge). Draws with worst λ_max(B) > 1.546 (= 1.6 / 1.035) over all
six conditions are discarded analytically; the survivors and the Stage-I
point are simulated in W1–W4 and W6.

## Stage II-b — local refinement

Around the current best candidate (the winner of the selection rule below,
applied to everything evaluated so far, without the settling check): 64
Sobol draws, seed 20260910 + r in round r = 1, 2, …, in a box with spans
f · (Stage II-a spans), f = 0.5 in round 1; screened and simulated as in
II-a. After each round the rule is applied again: if a new candidate wins,
the next box is centred on it with the same f; otherwise f is halved. The
stage ends when f would fall below 0.125, or after 4 rounds.

## Selection — over every candidate evaluated in II-a and II-b

1. **Filter** (W1–W4, W6): tap operations ≤ 30 per transformer and day;
   bank commits ≤ 12 per bank and day; no hunting of taps or banks
   (opposite moves of one device < 2 h apart with no new disturbance);
   AVR step ≤ 0.005 pu; no voltage outside the hard corridor 0.90–1.10 pu;
   no power-flow failure; λ_max(B) ≤ 1.6; interface-Q error ≤ 3 × that of
   the Stage-I point.
2. **Rank** on e_v(TS), the per-bus RMS deviation from the voltage reference
   in the TS; candidates within 2 % of the best are ordered by e_v(DS), the
   same in the STSs.
3. **Settling requirement**, checked in rank order on the isolated-STS step
   bank (120 steps, W1–W4, W6, steps capped at 10 Mvar): p95 of the settling
   iterations ≤ 3 N_inner = 27 and median residual ≤ 1 Mvar. A candidate that
   fails is removed and the ranking recomputed among the rest.

The first candidate that passes is the result, written as
`configs/paramsets/ch9_selected_band_2026_10.json`.

## Execution notes

- A fresh application: no simulation of an earlier campaign is reused, since
  the operating band changes the closed loop.
- 38 workers for Stage II-a, 30 for Stage II-b and the settling checks.

## Execution record (results only; the procedure above is unchanged)

- 2026-10-25 21:54 — Driver started (code state `code_state/`, HEAD 9d03c34);
  configuration check: DSO band (0.92, 1.08), banks `miqp`.
- 2026-10-26 10:06 — Stage II-a: 119 survivors of the screen plus the Stage-I
  point simulated in W1–W4, W6; 47 / 120 feasible, best faf088d2570e.
- Stage II-b, f = 0.5 throughout (every round found a new best, so the stage
  ended at the 4-round cap, not by the span rule):

  | round | centre | survivors / 64 | new best | e_v(TS) | e_v(DS) |
  |---|---|---|---|---|---|
  | 1 | faf088d2570e | 32 | e552b328fd76 | 0.00507 | 0.00607 |
  | 2 | e552b328fd76 | 35 | ff3f747d3f96 | 0.00471 | 0.00796 |
  | 3 | ff3f747d3f96 | 38 | c08b1d3d33ab | 0.00449 | 0.00991 |
  | 4 | c08b1d3d33ab | 17 | fde2e81af6b1 | 0.00440 | 0.01305 |

- 2026-10-27 05:22 — Selection over all 242 evaluated candidates: 94
  feasible. Ranking 1. fde2e81af6b1, 2. c08b1d3d33ab (e_v(TS) 0.00449, just
  outside the 2 % band of 0.004488), 3. e27ce5e4f01e, 4. 301c285ee52b; the
  four settling checks run in parallel (4 × 7 workers).
- 2026-10-27 08:51 — The settling run of 301c285ee52b (rank 4) was aborted at
  47 / 120 steps on the author's request, to free computing capacity; it is
  logged as FAIL with `error: no steps.csv`, which is **not a measured
  failure** (`expC/ABORTED_cand_301c285ee52b.txt`). It has no bearing on the
  result, since rank 1 passed.
- 2026-10-27 11:27 — Settling: fde2e81af6b1 PASS (p95 25.0; per DSO 24.0 /
  17.55 / 25.55 / 17.75; 4 / 120 steps > 27; max 33; median residual
  0.35 Mvar), c08b1d3d33ab PASS (22.05, 0.25), e27ce5e4f01e PASS (22.05, 0.24).
- **2026-10-27 11:58 — RESULT: `configs/paramsets/ch9_selected_band_2026_10.json`
  = fde2e81af6b1.** Over W1–W4, W6 (Stage-I point in brackets): e_v(TS)
  0.00440 (0.00547), e_v(DS) 0.01305 (0.00749), e_q 299 762 (1 039 596),
  λ_max(B) 1.467; taps ≤ 4 (TS) / ≤ 2 (STS) per transformer and day, bank
  commits ≤ 4 per bank and day, no hunting of taps or banks, AVR step
  ≤ 0.0035 pu, no bus outside 0.90–1.10 pu, no power-flow failure. Weights:
  g_q 0.1738, dso_g_v 659.9, g_w_der 0.03493, g_w_pcc 0.06768, g_w_gen
  7.277·10⁵, g_w_tso_oltc 25.59, g_w_tso_shunt 23.81, g_w_dso_der 0.708,
  g_w_dso_oltc 5.834. W5 (report only, in neither filter nor ranking):
  e_v(TS) 0.00779 (0.01097), e_v(DS) 0.01172 (0.00973). Driver total 38.1 h.
