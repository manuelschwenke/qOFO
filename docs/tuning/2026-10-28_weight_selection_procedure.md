# Controller weight selection: the procedure (Ch 8, Experiment B)

**Status:** FROZEN 2026-10-28 before its application, which is run 0008.
The rules were fixed by commit dddfa45 (procedure 2026-10-25 plus the
declared bound) before the run. This consolidated text was written during
the run and changes none of them. Applied once; its output is the weight set
used in Part III, whatever it is.

**Execution:**
`results/ch_09_parameter_selection/0008_2026-10-28_103321_ch_9_3_amendment1/`
(`experiments/ch_09_parameter_selection/ch_9_3_amendment1.py`). Its
Stage-II units come from
`results/ch_09_parameter_selection/0001_2026-10-25_215433_weight_selection_driver_band/`
(see *Execution notes*).

**Development history** (not part of the procedure). The procedure was
developed on preliminary campaigns, recorded in:
- `2026-10-16_protocol_experiment_B_rerun.md`;
- `2026-10-22_protocol_experiment_B_banks.md`;
- `2026-10-24_weight_selection_procedure.md`;
- `2026-10-25_weight_selection_procedure.md`, applied in run 0001, without
  the engagement bound of the filter. The bound was declared in
  `2026-10-28_amendment1_sts_engagement_bound.md`.

This document supersedes them.

## Configuration

**Network.** `rural_700` with DSO_3 ×2 (DER and active load).

**Banks.** MSR on the first and MSC on the second coupling-transformer
tertiary, 2 × 25 Mvar each. They are dispatched as integer inputs of the
TS-MIQP, with MSC/MSR exclusion per STS and a holding time of 1800 s.

**Controller settings.**
- tap lock 5 iterations;
- bank V-rescaling on;
- T_TS 180 s, T_STS 20 s, dt 20 s;
- measurement noise as `make_config()` (seed 20260717).

**Voltage limits.** **Every DSO controller holds its observed buses in an
operating band of 0.92–1.08 pu**, a margin of 0.02 pu to the grid-code
limits. The band is enforced as a soft output constraint priced by the
voltage slack weight (10⁹). The TS controllers keep 0.90–1.10 pu.

**Windows.** Bank of windows W1–W6 (`tuning.scenarios_mc_v2.CONDITIONS`),
one disturbance each, 12 h. W5 has no DER reactive capability at either
level. It enters neither the filter nor the ranking and is simulated only for
the selected weights, for the report.

## Stage I — analytic starting point

- **Gauge:** g_v = 4·10⁴ (TS voltage priority 1 at σ = 0.005 pu).
- **Priorities:** π_q = π_v,DS = 1, τ = 1.
- **Continuous step prices:** each layer's prices are scaled so that the
  worst area and condition has λ_max(B) = 1. B carries one bank column per
  STS (MSC/MSR exclusion).
- **Integer step prices:** from the commit rule
  g_w = 2 g_V |Σ_j H_V[j,i]| Δv_eng, with Δv_eng = 0.0075 pu (TS taps),
  0.015 pu (STS taps) and 0.010 pu (MSC/MSR banks).

Result: `configs/paramsets/stage1_start_10_2026_banks.json`. The operating
band does not enter H or B, so the point is the same as without it.

## Stage II-a — global screening

- 512 scrambled Sobol draws, seed 20260910.
- Log box centred on the Stage-I point, with spans of:
  - ±1.5 decades for g_q and dso_g_v;
  - ±1.0 for g_w_der, g_w_pcc, g_w_dso_der and g_w_gen;
  - ±0.5 for g_w_tso_oltc, g_w_dso_oltc and g_w_tso_shunt.
- g_v is fixed (gauge).
- Draws with worst λ_max(B) > 1.546 (= 1.6 / 1.035) over all six conditions
  are discarded analytically.
- The survivors and the Stage-I point are simulated in W1–W4 and W6.

## Stage II-b — local refinement

Each round r = 1, 2, … works around the current best candidate, i.e. the
winner of the selection rule below applied to everything evaluated so far,
without the settling check:
- 64 Sobol draws, seed 20260910 + r;
- a box with spans f · (Stage II-a spans), f = 0.5 in round 1;
- screened and simulated as in II-a.

After each round the rule is applied again:
- if a new candidate wins, the next box is centred on it with the same f;
- otherwise f is halved.

The stage ends when f would fall below 0.125, or after 4 rounds.

## Selection — over every candidate evaluated in II-a and II-b

1. **Filter** (W1–W4, W6). A candidate is admitted only if all of the
   following hold:
   - tap operations ≤ 30 per transformer and day;
   - bank commits ≤ 12 per bank and day;
   - no hunting of taps or banks (opposite moves of one device < 2 h apart
     with no new disturbance);
   - AVR step ≤ 0.005 pu;
   - no voltage outside the hard corridor 0.90–1.10 pu;
   - no power-flow failure;
   - λ_max(B) ≤ 1.6;
   - interface-Q error ≤ 3 × that of the Stage-I point;
   - **the strongest STS coupler engages inside the DSO operating band:**

     $$
     \theta_{\mathrm{STS,min}} = \min_{c \in \{W1..W6\}} \min_i
     \frac{g_{w,\mathrm{dso\_oltc}}}{2\, g_{V,\mathrm{DS}}\,|\sum_j H_V^{(c)}[j,i]|}
     \le V_{\max,\mathrm{band}} - V_\mathrm{ref} = 0.05\ \mathrm{pu}
     $$

     H_V is that of the Stage-I operating points, and the rule is evaluated
     with the Stage-I commit-rule machinery. Above this bound no coupler can
     commit a step on the voltage gradient while the STS is inside its band.
2. **Rank** on e_v(TS), the per-bus RMS deviation from the voltage reference
   in the TS. Candidates within 2 % of the best are ordered by e_v(DS), the
   same in the STSs.
3. **Settling requirement**, checked in rank order on the isolated-STS step
   bank (120 steps, W1–W4, W6, steps capped at 10 Mvar): p95 of the settling
   iterations ≤ 3 N_inner = 27, and median residual ≤ 1 Mvar. A candidate that
   fails is removed and the ranking recomputed among the rest.

The first candidate that passes is the result, written as
`configs/paramsets/ch9_selected_band_2026_10_a1.json`.

## Execution notes

- **Reuse of Stage-II units.** Stage II is deterministic: seeded draws, a
  deterministic screen and bit-exactly reproducible simulations. Its
  (candidate, window) units are taken from run 0001, for two reasons:
  - The decision path of this procedure coincides with that of run 0001 at
    every round (same incumbents after II-a and rounds 1–3, f = 0.5
    throughout, end at the 4-round cap;
    `analysis/tap_engagement/replay_path.py`). The evaluated set is
    therefore the same 242 candidates.
  - In run 0008, 20 units (four weight sets × W1–W4, W6) were re-simulated
    and reproduce the stored e_v(TS) exactly.
- **Settling results.** Settling results measured in run 0001 are reused for
  the same reason. A run that was aborted there is repeated.
- **Workers.** 8 workers for the settling check.

## Execution record (results only; the procedure above is unchanged)

- **2026-10-28 10:33 — Run 0008 started** (commit dddfa45).
  - 140 / 242 candidates satisfy the engagement bound, 94 are feasible
    without it, and 51 under the full filter.
  - Ranking: 1. 301c285ee52b (θ_STS,min 0.0147 pu), 2. c08b1d3d33ab
    (settling PASS in run 0001), 3. 67ad1cf843ac, 4. 9e48320e8a01.
- **Settling of 301c285ee52b:** running.
