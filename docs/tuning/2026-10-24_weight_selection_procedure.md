# Controller weight selection: the procedure (Ch 8, Experiment B)

**Status:** FROZEN 2026-10-24 before its application. Applied once; its
output is the weight set used in Part III, whatever it is. Execution:
`results/THESIS_ch8_weight_selection_20261024/` (`driver.py`, `driver.log`).

Development history (not part of the procedure): the procedure was developed
on preliminary campaigns, recorded in `2026-10-16_protocol_experiment_B_rerun.md`
and `2026-10-22_protocol_experiment_B_banks.md`.

## Configuration

Network `rural_700` with DSO_3 ×2 (DER and active load); MSR on the first,
MSC on the second coupling-transformer tertiary, 2 × 25 Mvar each, dispatched
as integer inputs of the TS-MIQP with MSC/MSR exclusion per STS and a holding
time of 1800 s; tap lock 5 iterations; bank V-rescaling on; T_TS 180 s,
T_STS 20 s, dt 20 s; measurement noise as `make_config()` (seed 20260717).
Bank of windows W1–W6 (`tuning.scenarios_mc_v2.CONDITIONS`), one disturbance
each, 12 h; W5 has no DER reactive capability at either level and is
simulated but excluded from ranking and filter.

## Stage I — analytic starting point

Gauge g_v = 4·10⁴ (TS voltage priority 1 at σ = 0.005 pu); priorities
π_q = π_v,DS = 1, τ = 1; each layer's continuous step prices scaled so that
the worst area and condition has λ_max(B) = 1, with one bank column per STS
in B (MSC/MSR exclusion); integer step prices from the commit rule
g_w = 2 g_V |Σ_j H_V[j,i]| Δv_eng with Δv_eng = 0.0075 pu (TS taps),
0.015 pu (STS taps), 0.010 pu (MSC/MSR banks).
Result: `configs/paramsets/stage1_start_10_2026_banks.json`.

## Stage II-a — global screening

512 scrambled Sobol draws, seed 20260910, in a log box centred on the
Stage-I point, spans ±1.5 decades (g_q, dso_g_v), ±1.0 (g_w_der, g_w_pcc,
g_w_dso_der, g_w_gen), ±0.5 (g_w_tso_oltc, g_w_dso_oltc, g_w_tso_shunt);
g_v fixed (gauge). Draws with worst λ_max(B) > 1.546 (= 1.6 / 1.035) are
discarded analytically; the survivors and the Stage-I point are simulated
in every window.

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
   AVR step ≤ 0.005 pu; no voltage outside the hard corridor; no power-flow
   failure; λ_max(B) ≤ 1.6; interface-Q error ≤ 3 × that of the Stage-I
   point.
2. **Rank** on e_v(TS), the per-bus RMS deviation from the voltage reference
   in the TS; candidates within 2 % of the best are ordered by e_v(DS), the
   same in the STSs.
3. **Settling requirement**, checked in rank order on the isolated-STS step
   bank (120 steps, W1–W4, W6, steps capped at 10 Mvar): p95 of the settling
   iterations ≤ 3 N_inner = 27 and median residual ≤ 1 Mvar. A candidate that
   fails is removed and the ranking recomputed among the rest.

The first candidate that passes is the result, written as
`configs/paramsets/ch9_selected_final_2026_10.json`.

## Execution notes

- Stage II-a is deterministic (fixed seeds, fixed code and configuration).
  Its records from 2026-10-22/23 (`results/THESIS_ch8_expB_banks_20261022/bank12h`)
  are reused, conditional on a reproduction check run first: two of its
  windows re-simulated (the Stage-I point in W1; its selection-rule winner
  in W5, which has bank commits and tap hunting) must reproduce every
  stored metric exactly. If they do not, Stage II-a is re-run.
- Settling results already measured with the same code for candidates in the
  set are reused for the same reason.
- 30 workers; Stage II-b ≤ 4 rounds × ~4 h; settling checks ~1.5–3 h.
