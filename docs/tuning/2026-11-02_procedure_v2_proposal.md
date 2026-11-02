# Weight-selection procedure v2 (proposal): design the subordinate layer first

**Status:** PROPOSAL for discussion with the author, 2026-11-02. Nothing is implemented,
nothing is frozen. Once the author decides, the chosen text is frozen before the first
simulation of the new campaign, as with the procedure of 2026-10-28.

## 1. What procedure v1 produced on the commitment base (facts)

- **Runs:**
  - Run `0001_2026-10-31_135000_ch_9_6_full_procedure` (amendment 2).
  - The re-selection `0003_2026-11-01_203826_ch_9_7_amendment3` (amendment 3, θ band
    0.02–0.04 pu).
- **Settling checks** (isolated-STS step bank, 120 steps, p95 n_settle ≤ 27 and median
  residual ≤ 1 Mvar):
  - 8 in run 0001;
  - 19 in run 0003 (batches 1–4), plus two diagnostics taken over (d162c73ca1fc,
    e63aed2d04cb).
  - **Result: 29 checks, one pass, d162c73ca1fc,** with p95 26 of 27 and residual
    0.71 Mvar.
- **Probe:** c08b1d3d33ab also passes (p95 21, 0.27 Mvar), but it is inadmissible: θ 0.0195
  is below the band, and it fails the hunting limit.
- **Run 0003 ended after batch 4.**
  - The author decided at 08:39 that batch 5 is not run.
  - The driver ended at 08:57 on a shard-log pull error (WinError 32, fixed since), after
    all six batch-4 checks had finished.
  - The batch-4 verdicts were computed read-only (`expC/batch4_verdicts_readonly.json`):
    all FAIL.
  - Batch 5 would have checked e738, 6cb6, d088, 17cd and f7ce, with predicted residuals
    of 1.7–2.2 Mvar.
  - Had all five failed, d162 would have been selected.

## 2. Diagnosis

### 2.1 The settling requirement is a property of the STS layer alone (model facts)

- **The TS weights do not enter the test.** The isolated-STS test freezes the TS
  controller; its weights only affect the t = 0 solve (`ch_8_D1_settling_screen.py`).
- **The STS weights reduce to three ratios.**
  - Each STS MIQP is invariant to a common scaling of its own weights (g_q, dso_g_v,
    g_w_dso_der, g_w_dso_oltc). The only exception is the fixed voltage-slack price.
  - The STS layer therefore has three effective degrees of freedom:
    - ρ = dso_g_v / g_q sets the steady-state offset of a Q step;
    - κ = g_q / g_w_dso_der sets the convergence speed;
    - θ_STS,min ∝ g_w_dso_oltc / dso_g_v sets where the couplers commit.
- **The TS layer does not use g_q.** It prices its interface-Q rows with `tso_g_q_pcc`
  (= 0); `stage1_screen._layer_weights`.

### 2.2 v1 searches the STS ratios jointly with the TS weights, and its centre lies outside the STS requirements

- **The Stage-I centre is infeasible.** v1's Stage I (π_q = π_v,DS = 1, λ_STS = 1,
  Δv_eng,STS = 0.015 pu) puts the centre at ρ = 2.5·10⁵ and θ = 0.015. That lies outside
  the θ band and is predicted to fail settling (calibrated residual 2.2 Mvar).
- **The feasible share of the 9-D II-a box is tiny** (II-a draws of run 0001, same seed;
  θ on the run's operating points; settling by the calibrated predictor of § 3):

  | requirement | draws (of 512) |
  |:--|--:|
  | λ_max(B) screen | 115 (22.5 %) |
  | θ band 0.02–0.04 | 49 (9.6 %) |
  | predicted settling | 23 (4.5 %) |
  | θ band and settling | 3 (0.6 %) |
  | λ, θ and settling, with g_w_tso_oltc ≥ 25 (hunting-free) | 1 (0.2 %) |

- **II-b refines around the ranking leader, without settling.** The e_v(DS) tie-break
  favours large ρ: Spearman −0.66 over the 85 feasible candidates of run 0001. All II-b
  centres lie at ρ ≥ 2.8·10⁵: c753 2.8·10⁵, dd14 3.1·10⁵, 068e 5.2·10⁵.
- **So the failure is structural, not chance.** Run 0001 simulated 244 candidates, and only
  4 of its 45 low-ρ candidates came from II-b.

## 3. Linear settling predictor: validation on the commitment base (facts)

- **Model.** `tuning/settling_predictor.py` (2026-09-29) is a gradient-step model of the
  STS update on the controllers' cached sensitivities. It ignores integer actuators,
  capability limits, noise and nonlinearity, and needs no simulation.
- **Data:** 30 measured checks on this base (runs 0001 and 0003, diagnostics, c08b).

  | | Spearman (measured vs predicted) | calibration |
  |:--|--:|:--|
  | median residual | 0.985 | measured ≈ 1.7 · predicted^0.60 (the predictor is optimistic) |
  | p95 n_settle | 0.952 | measured − predicted ∈ [−1, +5]; the largest offsets are at the passing end (c08b +5, d162 +3) |

- **Files:** `results/diagnostics/2026-11-02_settling_predictor_uc/` (`validation.csv`,
  `all_candidates.csv`, sensitivity cache).
- **These are tuning data:** the step bank in W1–W4 and W6, the same test as the
  procedure's. W5 and the Case-2/MC scenarios are not touched.

## 4. Options

**A. Minimal adaptation (keep the 9-D joint search).**
- Changes:
  - Stage I re-centred: Δv_eng,STS = 0.03; π_q and λ_STS as in § 5.
  - The θ band enters the filter from the start.
  - New filter: predicted settling (calibrated, with margin). The II-b centres then
    respect it.
  - The final settling loop is unchanged.
- **For:** the smallest change to the frozen text.
- **Against:**
  - Search effort goes into STS dimensions the TS ranking is indifferent to (e_v(TS)
    vs ρ: Spearman −0.03).
  - The e_v(DS) tie-break pushes candidates to the edge of the predicted-feasible region,
    so the final check rests on the calibration margin.
  - Runtime stays open-ended (sequential checks).

**B. Cascade design (recommended): the subordinate layer first, then the supervisory layer.**
- Changes:
  - Stage I designs the STS weights from their own specification and verifies them once
    by simulation.
  - Stage II searches only the five TS weights (g_w_der, g_w_pcc, g_w_gen, g_w_tso_oltc,
    g_w_tso_shunt).
- **For:**
  - Settling and θ are met by construction and verified before the search. The TS search
    cannot change the isolated test.
  - 5 instead of 9 dimensions: about 3 Sobol points per dimension with 256 draws, against
    2 per dimension with 512 in 9-D.
  - The open-ended settling loop disappears, so runtime is predictable.
  - It follows the inner-loop-first rule of cascade control, matching the layer hierarchy
    and timescale separation of the thesis.
- **Against:**
  - The STS ratios are no longer optimised for e_v(TS) or e_v(DS). The data say e_v(TS)
    is indifferent to ρ; e_v(DS) will be worse than at the v1 leaders.
  - It is a methodological change (v2), not an amendment.

**C. Settling-aware II-b with simulated checks in every round.** Rejected: about 3 h per
check batch per round, and the same exhaustion risk.

## 5. Proposed v2 (option B) in detail

1. **Stage I-STS (new; analytic, then one verification).**
   - θ target 0.03 pu, the centre of the amendment-3 band: Δv_eng,STS = 0.03. Taps commit
     at 1.06 pu under a uniform deviation.
   - λ_STS target 1.4 instead of 1.0, for speed. c08b (λ 1.41) passes with p95 21.
     The filter ceiling stays at 1.6.
   - π_q (with π_v,DS = 1): the smallest priority, i.e. the largest ρ, whose calibrated
     prediction meets residual ≤ 0.5 Mvar and predicted p95 + 5 ≤ 27.
   - Design curve (λ_STS 1.4, θ 0.03; commitment-base operating points):

     | π_q | ρ | κ | predicted residual → calibrated (Mvar) | predicted p95 (+5 worst offset) |
     |--:|--:|--:|:--|:--|
     | 10 | 2.5·10⁴ | 0.277 | 0.18 → 0.60 | 18 (23) |
     | 20 | 1.25·10⁴ | 0.293 | 0.11 → 0.44 | 17 (22) |
     | 30 | 8.3·10³ | 0.298 | 0.07 → 0.36 | 16 (21) |

   - The rule selects π_q = 20. v1 had π_q = 1: ρ 2.5·10⁵, calibrated residual 2.2 Mvar.
   - **Verification:** π_q 10, 20 and 30 in parallel (one batch, about 3 h), with the
     unchanged criterion.
     - Declared choice: the largest ρ that passes.
     - If none passes: stop and report. No search on an unverified STS design.
2. **Stage I-TS: unchanged** (gauge g_v = 4·10⁴, λ_TS = 1, Δv_eng,TS 0.0075, banks 0.010,
   g_w_gen 10⁸, τ = 1). The TS weights are identical to v1's Stage I.
3. **II-a:** 256 Sobol draws over the five TS weights, with the v1 spans; the STS weights
   are fixed.
   - The λ screen is unchanged; 141 / 256 pass (55 %).
4. **II-b: unchanged** (64 draws per round, f = 0.5 with halving, at most 4 rounds; the
   centre is the winner of the selection rule).
5. **Selection: unchanged filter**, including hunting tolerance 0. θ is in band by
   construction, and the check stays. Rank on e_v(TS), with the 2 % band on e_v(DS).
6. **Settling:** one confirmation check of the winner (same criterion).
   - It differs from the Stage I-STS verification only through the t = 0 solve.
   - The v1 loop stays as a safety rule if it fails.
7. **W5 report**, unchanged.

## 6. Decisions needed before the campaign

1. Option A or B.
2. The π_q rule and its margins: calibrated residual ≤ 0.5 Mvar; predicted p95 + 5 ≤ 27.
3. The λ_STS target: 1.4 (proposed), against 1.0 or 1.2.
4. N_2A: 256 (proposed), against 512 (about 10 h more).
5. **The DSO operating band lower edge:** 0.92 as coded, or 0.98 from the author's
   1.03 ± 0.05. It enters every simulation, so it must be fixed first.
6. Δv_eng,TS: unchanged at 0.0075, or 0.010, which centres g_w_tso_oltc at 28, on the
   hunting-free side.
7. Hosts: 30 workers on e5wimi and 30 on e5power for about 1.5 days.

## 7. Expected cost (option B; 60 workers; run-0001 throughput of 1.2 units per worker-hour)

| step | units | time |
|:--|:--|:--|
| implementation, tests, dry run | — | 2–3 h |
| Stage I-STS verification (3 checks in parallel) | 3 × 120 steps | ≈3 h |
| II-a, 141 candidates × 5 windows | ≈700 | ≈10 h |
| II-b, 4 rounds × ≈35 candidates × 5 windows | ≈700 | ≈10 h |
| selection, confirmation check, W5 | 1 check + 1 window | ≈4 h |
| **total** | | **≈29 h** |

## 8. Risks and open points

- **Calibration.** The predictor's calibration rests on 30 checks, with only two passes at
  the low-ρ end. The Stage I-STS verification covers this before any search.
- **The e_q filter** is "≤ 3 × the Stage-I point". It becomes stricter, because the new
  Stage-I point tracks well. TS candidates that move the PCC setpoints aggressively may be
  rejected. To be watched in II-a.
- **Hypothesis: e_v(TS) does not depend on κ either.** It is untested; κ varied jointly
  with ρ in v1.
- **Reporting.** The thesis reports v1, its failure on the commitment base and this
  diagnosis, and states v2 as declared before campaign 2.
