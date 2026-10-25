# Controller weight selection: the procedure (Ch 8, Experiment B)

**Status:** FROZEN 2026-10-25 before its application. Applied once; its
output is the weight set used in Part III, whatever it is. Execution:
`results/THESIS_ch8_weight_selection_band_20261025/` (`driver.py`,
`driver.log`).

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
