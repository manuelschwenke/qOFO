# Protocol: re-run of Experiment B (weight selection) on the final configuration

**Status:** FROZEN 2026-10-16 16:30 for execution under the author's delegation ("run all the reruns from the tuning campaign you need to run", 2026-10-16). D1–D6, D8–D10 by the author; D7 and the §9.2 procedure set by Claude under that delegation (marked, for the author to confirm). Code state recorded as a snapshot, not committed (§2). Execution: `results/THESIS_ch8_expB_10_2026/`. Once filled in, the protocol is frozen; any later deviation is recorded in §10 with date and reason, never by editing §1–§9.

**Claim this protocol supports (thesis, Ch 8 `ch:param:experiment:d`):** the controller weights used in Part III were obtained by the procedure described in Ch 8 — analytical starting weights, empirical selection by stated limits and a stated ranking — with every rule and input fixed before the run, applied once on the final configuration, and adopted as they came out.

---

## 1. Why a re-run is needed (established facts)

| Fact | Evidence |
|---|---|
| The selected `ch9_selected` (399e30d3b60b, 2026-09-16) was chosen on a configuration that no longer exists: tap lock 1 iteration in the campaign (now 5), no shunt voltage rescaling (now on), MSC and MSR on the same tertiary (now MSC on the second coupler), `shunt_int_g_w` 0.03. | `tuning/parameters.py` FIXED_OVERRIDES comment; daily logs 2026-10-14 (tap lock, rescaling), 2026-10-15 (placement) |
| Under today's configuration the Ch 8 ranking (e_v(TS) first) no longer prefers it: lowering only `g_w_pcc` to 27 / 13.9 × `g_w_der` lowers e_v(TS) by 4.4 % / 6.7 %. | `results/THESIS_pcc_ratio_sweep_20261015`, log 2026-10-15 |
| The selected point lies at the corner of the search box in the allocation direction: `g_w_der` −0.94 of ±1.0 decade, `g_w_pcc` +0.97 of ±1.0, `g_w_gen` −0.81 of ±1.0 from the starting weights. The ratio `g_w_pcc/g_w_der` = 81 is set largely by the box. | same log, "Added 2026-10-16" |
| Since the selection, three parameters were set by hand: `shunt_int_g_w` 0.06, `g_w_pcc` 27 × `g_w_der`, per-area `g_q` for DSO_1–3. | parameter sets `ch9_selected_shunt_gw006`, `_pcc27`, `_pcc27_gqceil`, `_pcc27_gq125` |
| Stage I has changed: the tap-engagement rule no longer subtracts ‖a_i‖² (2026-10-14), so the starting tap prices change. | daily log `2026-10-14_tap_engagement_rule_fix.md` |
| Both ranking metrics are **area-mean** deviations: e_v(TS) = RMS over zones of (zone-mean V − 1.03), e_v(DS) = RMS over STS of (area-mean V − 1.03) (`experiments/helpers/comparison_metrics.py::_rmsd_from_means`). The thesis text says "root-mean-square voltage deviation", and the controllers minimise per-bus deviations. Per-bus STS voltages are not recorded. | code; Ch 7 comment at `ch:setup:metrics:contribution` already asks for this check |

Precedent for re-running on a changed configuration: 2026-09-16, network change to DSO_3 ×2 (archived shortlist re-simulated, 168 units, 5.4 h).

## 2. Configuration frozen before the run

| Item | Value | Source |
|---|---|---|
| Network | `rural_700`, DSO_3 ×2 (DER and load) | `_ch9_selected_design.FROZEN_*` |
| Shunt placement | MSR on coupler 1 tertiary, MSC on coupler 2 tertiary, 2 × 25 Mvar each | `tso_shunt_m{sr,sc}_coupling_pos = 0 / 1` |
| Tap lock | 5 iterations of the owning controller; wall-clock OFO lock off | `int_cooldown = 5`, `ofo_oltc_cooldown_s = 0` |
| Shunt rescaling | on | `shunt_int_v_rescale = True` |
| Periods | T_TS = 180 s, T_STS = 20 s, dt = 20 s | Ch 8 |
| Shunt settings | g_w,sh from §3.2; δ = 10 Mvar; dwell 30 min; daily budget off | Ch 3 / Ch 4 |
| Measurement noise | as `make_config()` (profile "minimum", seed 20260717) | config |
| Window bank | W1–W6, evented, 12 h each; W5 run but excluded from the ranking (zero capability) | Stage-II evaluator |
| Code state | Not committed (no author go-ahead for a commit). Recorded instead in `results/THESIS_ch8_expB_10_2026/code_state/`: HEAD `781a81907332520bd89a0e1d4d570ded15107e7e` (branch `weights-reset-2026-09-10`), byte-exact `tracked_changes.patch` (verified with `git apply --check --reverse`), `status.txt`, copies of the 20 untracked source files. Captured 2026-10-16 16:35, before any campaign run. | snapshot |

## 3. Rules fixed before the run

### 3.1 Stage I — analytical starting weights (Ch 8, unchanged method)
- Gauge `g_v` = 4·10⁴; priorities π_v,TS = π_q = π_v,DS = 1; τ = 1.
- Movement weights proportional to ‖a_j‖², one scale per layer bisected to λ_max(B) = 1.0 on the worst area and condition.
- Tap prices from the corrected engagement rule `g_w = 2 g_V |Σ H_V| Δv_eng`, Δv_eng TS = 0.010 pu, STS = 0.020 pu (**D7**).
- Operating points: the six conditions on the frozen network (`results/stage1/operating_points`, key `net190b46e8`).

### 3.2 Shunt integrator weight (rule, not searched)
`g_w,sh = T_p · |g_H|_p90 / (2 (q/2 + δ) T_TS)`, with T_p the required commit persistence at the p90 boundary pressure.
- T_p = **D2** (recommended 60 min → 0.059 at |g_H|_p90 = 0.133; 30 min gave 0.030).
- |g_H|_p90: keep the 2026-09-11 reference 0.133, or re-measure at the Stage-I weights on the frozen configuration (**D2b**). With MSC and MSR on different tertiaries each bank has its own pressure, so a re-measurement is the cleaner choice.

### 3.3 Stage II-a — screening
- Sobol, n = 512, scrambled, seed **D8** (recommended: the archived 20260910 for comparability).
- Box centred on the new Stage-I point, log spans `g_q` 1.5, `dso_g_v` 1.5, `g_w_dso_der` 1.0, `g_w_tso_oltc` 0.5, `g_w_dso_oltc` 0.5, and for `g_w_der`, `g_w_pcc`, `g_w_gen` either 1.0 (as before) or 1.5 (**D4**).
- Analytic screen λ_max(B) ≤ 1.5459 (= 1.6/1.035) in every area and condition; survivors simulated 3 h on W1–W6.
- II-a output is a feasibility map and a shortlist; no II-a ranking is quoted.

### 3.4 Stage II-b — 12 h evaluation
- Shortlist = baseline + Pareto front (e_v(TS), e_v(DS)) + top 14 on e_v(TS) + top 8 on e_v(DS) among II-a survivors that pass the II-a limits, deduplicated (rule of the 2026-09-11 re-run).
- Each candidate: 12 h on W1–W6, evented, frozen network.

### 3.5 Selection (Ch 8)
1. **Filter:** tap operations ≤ 30 /day per transformer; hunting 0 (reversal < 2 h not explained by an event); AVR step ≤ 0.005 pu; voltage-corridor violations 0; PF failures 0; λ_max(B) ≤ 1.6; e_q ≤ 3 × e_q(starting weights).
   - Optional additional limit **D3**: e_v(DS) ≤ (1 + x) · e_v(DS)(starting weights), x fixed now.
   - Shunt switching: reported, not bounded (2026-09-15 scope statement), or bounded (**D9**).
2. **Rank** on e_v(TS); candidates within 2 % of the best are ordered by e_v(DS).
3. e_q is bounded, never minimised.
4. Metric definitions **D1**: area-mean (as coded) or per-bus RMS (as the thesis text reads and as the controllers optimise). Per-bus needs one recording change (per-area per-bus RMS deviation stored in the record) before the run.
5. **Settling check (Experiment C, existing Ch 8 rule, stated here for completeness):** the top-ranked candidate is run on the isolated-STS step protocol (`ch_9_1_ninner_isolated_sts.py --max-step-mvar 10 --save-trajectories`, 120 steps, W1–W4, W6). It passes if p95(`n_settle_final_1mvar`) ≤ 3 N_inner = 27 and the median |q_final − q_set| ≤ 1 Mvar (criterion of 2026-09-30, `results/THESIS_ch8_D_settling_20260929/criterion_check.py`). If it fails, the next candidates in rank order are checked and the first that passes is selected (Ch 8: "the selection would be repeated with the requirement as an additional limit").

### 3.6 Post-selection steps
**None.** In particular no per-area `g_q` and no hand-set allocation ratio. One global `g_q`, as Ch 8 states ("the same values apply in every area"). Per-area `g_q` is a Ch 11 sensitivity study (§8), not part of the tuning.

## 4. What is reported (declared now)
- Selected weights, λ_max(B) and binding row, feasibility counts and reasons for rejection.
- Ranking aggregate over W1–W4, W6 and the W5 figures separately.
- Box-edge check: offset of every selected coordinate in fractions of its span; any coordinate above 90 % is stated in the thesis.
- Allocation figures: `g_w_pcc/g_w_der`, recruited flow against the starting weights (Ch 7 definition, base = starting weights), shunt commits and reversals.
- For orientation only, not part of the selection: the same aggregates for `ch9_selected` and `ch9_selected_pcc27` re-run on the frozen configuration.

## 5. Acceptance
The top-ranked candidate is adopted unchanged as `ch9_selected_10_2026` (name **D10**). It replaces `SELECTED_PARAMSET` and the default of `run_multi_system_ofo.py`. If no candidate passes the filter, the protocol stops and the failure is reported. Rules are not relaxed to obtain a result.

## 6. Downstream (decide now which results follow the new weights) — **D6**
| Experiment / chapter | Uses `ch9_selected` today | Re-run? |
|---|---|---|
| Ch 8 Experiment C (settling) | yes | recommended (the rule says: if the selected weights fail the settling requirement, the selection is repeated with it as a limit) |
| Ch 8 D1 / D2 (droop) | yes | |
| Ch 9 (SVR case) | yes | |
| Ch 10 (multi-system, MC) | yes | |
| Ch 11 (STS sensitivity) | yes ("weights of Ch 8") | yes, by construction |

## 7. Compute plan (estimate, 40 cores)
| Step | Units | Wall |
|---|---|---|
| Stage I + screen | — | minutes |
| II-a, ~90–120 survivors × 6 windows × 3 h | ~600 | ~5 h |
| II-b, ~30 candidates × 6 windows × 12 h | ~180 | ~3 h |
| Selection, reporting | — | minutes |
Driver hygiene: start workers only for missing units (2026-10-15 pool fault), persist per unit.

## 8. Ch 11 sensitivity study: per-area interface weight (not part of the tuning)
**Role in the argument.** Experiment B defines one basic parameter set with a common `g_q`, used in every experiment of Ch 8–11. The per-area interface weight is not a tuning coordinate of the transmission design: in the cascade each STS operator owns its controller and sets its own priority between following the TSO request and holding its own voltages. Ch 11 therefore treats per-area `g_q` as **operator heterogeneity** and measures what an individual operator's choice does to its own delivery and voltage cost, and to the other areas through the TS. The study does not feed back into the weights of Ch 8–10. If a refined set is recommended, it is named as such, justified by the trade-off curves, and used only from Ch 11 on.

**Question (fits Ch 11 "reliable" and "affordable"):** how does an STS operator's own priority between interface tracking and internal voltages trade the delivery of the requested reactive power against voltage cost inside the STS, and does the trade depend on the STS's network properties (Λ, x_eff, x_TS)?

**Design, at the weights of §5:**
- One area at a time (OAT): `g_q` of one STS × m, the others at the reference, for each of DSO_1–4. m ∈ {0.25, 0.5, 1, 1.25, area ceiling} (ceiling = largest m with that area's λ_max(B) ≤ 1.546); optionally one point between the ceiling and λ = 2 to show the closed-loop margin.
- Joint DSO_1–3 at the same m values, DSO_4 at the reference: tests additivity through the TS and is the deployable variant.
- W1–W6, 12 h; about 21 candidates, ~126 units, ~2 h.

**Measures per area:** voltage cost f_V^STS (Ch 7 `eq:setup:metrics:vcost`, per bus, needs the recording change of D1), interface tracking (RMS of Q_act − Q_set, also at the next TS step), delivered and recruited flow, TS voltage benefit, DSO tap and DER activity, λ_max(B).

**Output:** per area a trade-off curve f_V^STS against tracking error, parameterised by m, with the reference and the λ ceiling marked; cross-area effects of each OAT move. It is a front of the swept family, not a proof of Pareto optimality.

## 9. Decisions (author)

| # | Decision | Recommendation | Author (2026-10-16) |
|---|---|---|---|
| D1 | Ranking metrics: area-mean (as coded) or per-bus RMS (as the text reads) | per-bus, recording change first | **per-bus RMS, as the text reads** — definition in §9.1 |
| D2 | Shunt persistence T_p | 60 min | **60 min** |
| D2b | p90 pressure: keep 0.133 or re-measure on the frozen configuration | re-measure | **re-measure** — procedure in §9.2 |
| D3 | Additional limit on e_v(DS) (value fixed now) or none | none: keeps the Ch 8 rule unchanged; the 2 % band and the voltage corridor already bound the DS side | **none** |
| D4 | Box span for `g_w_der`, `g_w_pcc`, `g_w_gen` | 1.5 decades | **1.5 decades** (`--span g_w_der=1.5 --span g_w_pcc=1.5 --span g_w_gen=1.5`) |
| D6 | Downstream re-runs | at least Experiment C and Ch 11 | **all necessary re-runs** (Ch 8 C, D1, D2; Ch 9; Ch 10; Ch 11) |
| D7 | Engagement deviations Δv_eng TS / STS | see §9.3 | **TS 0.0075 / STS 0.015 pu** — set by Claude under the author's delegation (author proposed STS 0.015; TS value per §9.3); **author to confirm** |
| D8 | Sobol seed | 20260910 | **20260910** |
| D9 | Shunt switching bounded in the filter or reported only | reported only (scope statement), now with T_p = 60 min | **reported only** |
| D10 | Name of the result | `ch9_selected_v2` | **`ch9_selected_10_2026`** |

### 9.1 D1 — metric definition (implemented 2026-10-16)
Per window: e_v(TS) = time mean of the quadratic mean over TS zones of each zone's spatial RMS of (V_i − V_i^set) over its voltage-observed buses (`v_rms_ts`, from `zone_v_rms_err_pu`); e_v(DS) = the same over the four STS areas and their controllers' voltage buses against the controllers' own references (`v_rms_ds`, from the new record field `dso_group_v_rms_dev_pu`). Areas are weighted equally, as for the TS before. Over windows: quadratic mean (`tuning.stage2_search.aggregate`, unchanged). True plant values, not measurements. Stage II runs with `--v-metric bus`; the area-mean values stay in every record as `e_v_*_areamean`.

### 9.2 D2b — p90 pressure procedure (proposed; object before §2 is frozen)
At the Stage-I weights of this protocol, on the frozen configuration, W1–W6 evented 12 h, with bank commits disabled (`shunt_int_g_w` = 1e12, so the integrator observes but never moves). Record `zone_tso_shunt_g_h` (new field) at every TS iteration; |g_H|_p90 = 90th percentile of |g_H| pooled over all eight banks, all TS iterations and all six windows. Disabled commits measure the pressure a bank sees before it acts, which is what the persistence rule is about; W5 is included because the rule concerns the bank, not the ranking. 6 units, ~40 min.

### 9.3 D7 — evidence (corrected commit rule, frozen operating points)
Engagement deviation actually implied by the tap prices used so far (strongest column per area and condition, systematic reading):

| weights | TS taps engage at (median, min–max) | STS taps engage at (median, min–max) |
|---|---|---|
| Stage-I start, stated 0.010 / 0.020 (old rule) | 0.0081 (0.0073–0.035) | 0.0172 (0.0159–0.0179) |
| `ch9_selected` | 0.0069 (0.0062–0.030) | 0.0172 (0.0159–0.0180) |

Stage-I tap prices under the corrected rule: Δv 0.010 / 0.020 → `g_w_tso_oltc` 28.5, `g_w_dso_oltc` 22.2 (+38 % / +26 % against 20.7 / 17.6 used so far, i.e. later engagement than in every run so far); STS 0.015 → 16.6; TS 0.0075 → 21.4.

Reading: everything simulated so far engaged the STS taps at ~1.7 % and the TS taps at ~0.8 %, not at the stated 2 % / 1 %, because the old rule subtracted ‖a_i‖². Keeping the stated values with the corrected rule would make both tap classes engage later than before. Recommendation: **STS 0.015 pu** (slightly earlier than the ~1.7 % so far, in the direction the author observed) and **TS 0.0075 pu** (reproduces the TS tap price used so far); the Stage-II box (±0.5 decade on tap prices) refines both.

## 10. Amendments after freezing
(date — what changed — why — effect on the claim)

**Amendment 1 — 2026-10-19, author decision, after run 1 (§11) ended with no admissible candidate.**
- **D7 confirmed** unchanged (TS 0.0075 / STS 0.015 pu).
- **D2b changed:** the p90 pressure is pooled over the **ranked windows W1–W4, W6 only**; W5 (zero IBG capability) is excluded, as it is from the ranking. From the stored run-1 measurement (no new runs): p90 |g_H| = 0.3611 (9 640 samples) → `shunt_int_g_w` = 3600 · 0.3611 / (2 · 22.5 · 180) = **0.1605** (`d2b_pressure/summary_ranked_windows.json`). W4 still sets the upper tail (45 % of its samples above the pooled p90). This is chosen after seeing that the six-window pooling gave 0.570; recorded as such.
- **D4 reverted:** box spans back to `tuning.stage2_search.DEFAULT_SPANS` (g_q, dso_g_v 1.5; g_w_der, g_w_pcc, g_w_dso_der, g_w_gen 1.0; tap prices 0.5) — exactly the "0.5–1.5 decades" the thesis text (Ch 8) states. Reason: at n = 512 the widened box (3.4× the volume) under-sampled; the old optimum 399e30d3b60b lies inside the box but was not sampled (orientation runs, §11). This changes search coverage, not acceptance: limits, ranking rule and settling check are unchanged.
- Everything else unchanged (n = 512, seed 20260910, per-bus metric, shortlist rule, selection, Experiment C). Executed as **run 2** in `results/THESIS_ch8_expB_10_2026/run2/`; run 1 stays as recorded.
- Effect on the claim: the weights come from one Stage-II campaign as described in Ch 8; the protocol record shows the first attempt and the two changes.

**Amendment 3 — 2026-10-20 08:15, clarification, fixed while the settling result of `df17c0c32904` was still unknown.** §3.5 item 5 paraphrased the settling rule as "the first candidate in rank order that passes". Ch 8 states: "if they failed it, the selection would be repeated with the requirement as an additional limit". Applied as the text reads (author's principle for D1 and Amendment 2): candidates that fail the settling check are removed and the ranking (e_v(TS), 2 % band on e_v(DS)) is recomputed among the remaining feasible candidates, the band anchored at the best remaining one; the settling check is needed for every candidate that can win the recomputed ranking. Here: rank 1 failed → best remaining 99ea756282bf (0.00507), band ≤ 0.00517 contains df17c0c32904 (0.00510, e_v(DS) 0.00642 < 0.00796) → df17c0c32904 is selected if it passes the settling check, 99ea756282bf otherwise.

**Amendment 2 — 2026-10-19 ~13:45, author decision ("do B").** Every candidate that passes the convergence check (all 119 of run 2, incl. the starting weights) is simulated for 12 h in W1–W6 and the selection is made over all of them, as the Ch 8 text states ("These candidates, together with the starting weights, are simulated for 12 h in each window W1–W6"). The 3 h pre-screen and the shortlist rule (§3.3–3.4) are dropped from the selection path; their run-2 outputs stay as a record. Decided after seeing that the shortlist rule excluded the draw nearest the old optimum (§11); motivated by matching the stated method, and applied to all candidates alike. The 23 shortlisted candidates already running at 12 h are reused (same unit files).

## 11. Execution record (results, not amendments)
- 2026-10-16 16:40 — Stage I (`tuning.stage1_screen --design --frozen-network --d-engage-ts 0.0075 --d-engage-sts 0.015`): `g_w_der` = `g_w_pcc` = 0.052662, `g_w_dso_der` 0.50170, `g_w_dso_oltc` 16.6417, `g_w_tso_oltc` 21.3734, `g_q` 0.04, `dso_g_v` 1e4, `g_v` 4e4, `g_w_gen` 1e8; worst λ_max(B) 1.000. `stage1/design.json`.
- 2026-10-16 17:20 — D2b (§9.2, 6 × 12 h, commits disabled, 0 commits, 11 568 samples): pooled p90 |g_H| = **1.2828** → `shunt_int_g_w` = 3600 · 1.2828 / (2 · 22.5 · 180) = **0.5701**. Per-window p90: W1 0.074, W2 0.096, W3 0.097, **W4 1.395, W5 3.610**, W6 0.059. The pooled value is set by the stressed windows (W4 load ramp, W5 no IBG capability + generator trip); the quiet windows lie near the earlier reference 0.133. Consequence: ~10× the 0.06 in use — banks commit after ~1 h of W4/W5-level pressure and practically never at quiet-window pressure. Followed as frozen (the window choice of §9.2 is not changed after seeing the result); **flagged for the author** together with D7. `d2b_pressure/summary.json`.
- 2026-10-16 17:35 — `configs/paramsets/stage1_start_10_2026.json` written by `--write-paramset` (weights identical to `design.json`, `shunt_int_g_w` 0.5701, provenance carries d_engage and the network).
- 2026-10-16 17:37 — II-a dry run: 93 of 512 Sobol draws pass the analytic screen at 1.5459 (18.2 %); + baseline = 94 candidates × 6 windows × 3 h. II-a launched with 36 workers, `--v-metric bus`, spans 1.5 on `g_w_der`, `g_w_pcc`, `g_w_gen` (`screen/`, log `screen_IIa.log`).
- 2026-10-16 21:20 — II-a finished: 564 / 564 units ok, 3.74 h, 94 candidate records. Feasible on the short bank (no wear): 75 / 94; rejections mostly interface-Q error > 3 × baseline (13). Feasibility map `screen/feasibility_IIa.json` (a record; no II-a ranking is quoted).
- 2026-10-16 21:25 — Shortlist (`ch_9_3_shortlist.py --hours 3`): Pareto front 6, top 14 on e_v(TS), top 8 on e_v(DS), baseline → **23 candidates** (`shortlist.json`, roles in `shortlist.meta.json`).
- 2026-10-16 21:27 — II-b launched: 23 candidates × 6 windows × 12 h, evented, `--v-metric bus`, 36 workers (`refine/`, log `refine_IIb.log`).
- 2026-10-17 00:35 — II-b finished: 138 / 138 units ok, 3.20 h, 23 candidate records.
- 2026-10-17 00:40 — Selection (`ch_9_3_select.py --hours 12`, wear judged; `selection/selection_report.json`, `select.log`): **4 / 23 feasible**. Rejections: hunting 15 (TS transformer taps and DSO_4 coupling taps; many reversals ~18 min after the W1/W3 generator trip at 4.8 h, counted because no new event explains them), DS corridor violations 2 (2119 / 2405 records), interface-Q > 3× baseline 2. Ranking (e_v(TS), 2 % band on e_v(DS)): **1. the baseline aa72d30f29f1 (Stage-I point)**: e_v(TS) 0.00544, e_v(DS) 0.00807, e_q 1 190 287, λ 1.000, taps/day TS 4 / DS 8; 2. 2bee7f792aee (e_v(TS) +11.9 %, e_v(DS) −45 %); 3. a535eefef892 (+13.5 % / −50 %); 4. 22b0502f5a2a. The three non-baseline feasible candidates have `g_q` 1.1–1.4 decades below the start (interface barely tracked) and more expensive TS taps (+0.25 to +0.36 decades); every candidate with TS taps cheaper than the start hunts. 0 candidates dominate the baseline on both voltage criteria.
- 2026-10-17 00:50 — Experiment C (§3.5 item 5) launched on ranks 1–4 in parallel, 9 workers each (`expC/rank*`); the rule takes the first passing candidate in rank order. Prior evidence: the previous Stage-I point failed C (p95 32, median residual 2.27 Mvar).
- 2026-10-17 02:00–03:15 — Experiment C (`expC/criterion_check.py`, `criterion_results.json`): **rank 1 (baseline) FAIL** p95 32.05, median residual 2.32 Mvar; **rank 2 FAIL** p95 33.0, 5.68 Mvar; **rank 3 FAIL** (04:25) p95 33.0, 6.31 Mvar; **rank 4 FAIL** p95 35.0, 8.94 Mvar (5 censored).
- **2026-10-17 04:26 — PROTOCOL OUTCOME: no admissible candidate.** All four candidates that pass the filter fail the settling requirement; per §5 the protocol stops here and no parameter set `ch9_selected_10_2026` is written. No rule was relaxed. Any continuation requires an amendment (§10) decided by the author.
- 2026-10-17 03:30 — Orientation (§4, declared; `orientation/`, 12 h W1–W6 on this configuration incl. shunt 0.570, per-bus metric): `ch9_selected` (399e30d3b60b) **feasible**, e_v(TS) 0.00510, e_v(DS) 0.00685, e_q 118 097, hunting 0; `ch9_selected_pcc27` **feasible**, 0.00499 / 0.00750 / 163 211, hunting 0. Both dominate the selected baseline (0.00544 / 0.00807) on both voltage criteria; neither was in the search bank (the old point lies inside the new box but was not sampled by 512 draws in the widened 8-D box). The shunt weight is not what makes candidates hunt (these weights do not hunt at 0.570, as they did not at 0.06 in the 2026-10-15 sweep).
- 2026-10-17 03:45 — **Diagnostic, outside the frozen protocol, not used for any selection pending the author's decision:** the archived 28-candidate shortlist of the 2026-09-16 re-confirmation (`results/THESIS_ch9_3_stage2_dso3x2/refine_evented`) re-simulated on this configuration (precedent: 2026-09-16), `diag_archived_shortlist/`.
- 2026-10-17 08:56 — Diagnostic finished: 162 / 162 units ok, 4.72 h, 29 records. **Diagnostic union selection** (unchanged rule; new bank 23 + archived 28 + orientation, 52 records; `diag_union/selection_report.json`): 14 / 52 feasible. Ranked: 1. `80ba9e00271f` = `ch9_selected_pcc27` weights (e_v(TS) 0.00499, e_v(DS) 0.00750, e_q 163 211, λ 1.357) — **entered only via the orientation runs, i.e. a hand-made set, not a search result**; 2. `399e30d3b60b` = `ch9_selected` (0.00510, 0.00685, 118 097, λ 1.310; from the archived search shortlist); 3. 2f35cfb5086d; 4. the baseline. Restricted to search-generated candidates (new bank + archived shortlist) the rule selects **399e30d3b60b**. Pareto front (6): 80ba9e00271f, 399e30d3b60b, 59e88bcfa362, 1536fc132bde, 2bee7f792aee, a535eefef892.
- 2026-10-17 09:00 — Diagnostic Experiment C launched on 399e30d3b60b and 80ba9e00271f (18 workers each, `expC/diag_*`).
- 2026-10-17 11:05 — Diagnostic C, **80ba9e00271f (`pcc27` weights) PASS**: p95 23.05 (per DSO 20.55 / 17.2 / 25.1 / 12.1), 3 / 120 steps > 27, max 35, median residual 0.38 Mvar.
- 2026-10-17 11:10 — Diagnostic C, **399e30d3b60b (`ch9_selected` weights) PASS**: p95 23.05 (per DSO 20.0 / 15.1 / 25.55 / 11.55), 3 / 120 > 27, max 34, median residual 0.38 Mvar — practically its 2026-09-29 result (23.05; 20.0 / 14.65 / 25.55 / 11.55).
- **Run 2 (Amendment 1)** — 2026-10-19 09:20: code snapshot `run2/code_state/` (tracked code identical to run 1); start set `stage1_start_10_2026_r2` = run-1 Stage-I weights with `shunt_int_g_w` 0.16050; II-a dry run with the default box: 118 / 512 survivors (23.0 %; September: 116); II-a launched (`run2/screen/`).
- 2026-10-19 13:26 — Run 2 II-a finished: 714 / 714 ok (one pool restart, resumed from disk), 119 records; 88 / 119 feasible on the short bank. Shortlist 23 (Pareto 9). The draw nearest to the old optimum, `df17c0c32904` (0.039 decades from 399e30d3b60b), is feasible but ranks 22 / 88 on both criteria (cut-offs: 14th TS, 8th DS) and is therefore not shortlisted; not added by hand. II-b launched on the 23 (`run2/refine/`).
- Observation for the author: the Ch 8 text says all candidates that pass the convergence check are simulated for 12 h in W1–W6; it does not describe the 3 h pre-screen and shortlist used here (September practice). Evaluating all 119 at 12 h would match the text exactly; the 23 running now are a subset of that and would be reused.
- 2026-10-19 16:48 – 2026-10-20 04:31 — Run 2, all candidates at 12 h (Amendment 2): 582 new units ok (11.7 h; 138 reused), 120 records. One record (`ea44681c81ba`) is the starting weights a second time (written without the shunt field, hence another fingerprint; identical metrics) and was moved to `refine/duplicates_of_baseline/` before the selection.
- 2026-10-20 04:40 — Run 2 selection (`run2/selection/selection_report.json`): **45 / 119 feasible**. Ranked: 1. `f879539ceec5` e_v(TS) 0.00494, e_v(DS) 0.00773, e_q 771 509, λ 1.277, g_w_pcc/g_w_der 11.5; 2. `99ea756282bf` 0.00507 / 0.00796 / 232 973, λ 1.408, ratio 7.8; 3. `df17c0c32904` (old-optimum neighbour) 0.00510 / 0.00642 / 112 838, λ 1.260, ratio 81; 4. `c268ac6737ed` 0.00530 / 0.00572 / 239 924, ratio 13.2; 5. starting weights 0.00533 / 0.00784. Rank 1 alone within the 2 % band. Experiment C launched on ranks 1–4 (`run2/expC/`).
- 2026-10-20 06:00–08:22 — Run 2 Experiment C: rank 1 f879539ceec5 **FAIL** (p95 28.0, residual 1.51 Mvar); rank 2 99ea756282bf PASS (24.05, 0.61); rank 3 df17c0c32904 PASS (23.05, 0.38); rank 4 c268ac6737ed FAIL (29.0, 1.23).
- 2026-10-20 08:25 — **Repeated selection with the settling limit (Amendment 3)** (`run2/selection/selection_with_settling.json`): best remaining e_v(TS) 0.00507 (99ea756282bf); band ≤ 0.00518 = {99ea756282bf, df17c0c32904}, both with settling results; tie-break on e_v(DS) → **df17c0c32904 selected** (e_v(TS) 0.00510, e_v(DS) 0.00642, e_q 112 838, λ 1.260; vs the starting weights −4.5 % / −18.2 % / 0.10×).
- 2026-10-20 08:30 — **RESULT: `configs/paramsets/ch9_selected_10_2026.json`** = df17c0c32904 weights + `shunt_int_g_w` 0.1605. Every weight within ±5.4 % of the September selection 399e30d3b60b (distance 0.04 decades; g_w_pcc/g_w_der 81.1): the procedure re-finds the same optimum on the final configuration. Runner-up 99ea756282bf (ratio 7.8, also passes the settling check) is the strongly STS-using alternative within the 2 % band on e_v(TS). Not switched: `SELECTED_PARAMSET`, the `run_multi_system_ofo.py` default (author).
- **Status for the author's decision (run 1):** under an amendment "candidate set = union of the search-generated sets (this campaign + archived 2026-09-16 shortlist)", the unchanged rule selects 399e30d3b60b, which passes Experiment C → admissible; the set would be its nine weights + `shunt_int_g_w` 0.570 (frozen configuration). `pcc27` (80ba9e00271f) ranks above it and passes C, but entered by hand. Nothing written pending the decision.
