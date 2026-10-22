# Protocol drafted: re-run of Experiment B on the final configuration

- Timestamp: 2026-10-16 (CEST).
- Reason: Author wants to state in the thesis that the weights were found by the procedure of Ch 8 (Experiment B). The weights in use (`ch9_selected_pcc27_gqceil`) are `ch9_selected` plus three hand-set changes, and `ch9_selected` itself was selected on a configuration that no longer exists. Author asked for a protocol that fixes all rules and inputs before a re-run.

## What changed

| File | Change |
|---|---|
| `configs/paramsets/ch9_selected_pcc27_gq125.json` | New (see `2026-10-15_pcc_ratio_sweep.md`, "Added 2026-10-16"). |
| `docs/tuning/2026-10-16_protocol_experiment_B_rerun.md` | New. Draft protocol: reasons for the re-run (§1), frozen configuration (§2), rules for Stage I, the shunt weight, II-a, II-b and the selection (§3), no post-selection steps (§3.6), declared outputs (§4), acceptance as-is (§5), downstream decisions (§6), compute plan (§7), Ch 11 per-area `g_q` study as operator heterogeneity, not fed back into Ch 8–10 (§8), decision table D1–D10 (§9), amendment log (§10). |

No code changed.

## Facts established while drafting

- Both ranking metrics are area-mean deviations: `voltage_rmsd_ts` / `voltage_rmsd_ds` (`experiments/helpers/comparison_metrics.py`) take the RMS over zones / STS of (area-mean V − 1.03), via `_rmsd_from_means(records, "zone_v_mean" | "dso_group_v_mean_pu")`. An area with buses at 1.00 and 1.06 scores zero. The thesis text (Ch 8) says "root-mean-square voltage deviation"; the controllers minimise per-bus deviations. Per-bus STS voltages are not recorded (`bus_vm_pu` holds 30 entries); TS per-zone per-bus RMS exists as `zone_v_rms_err_pu`. → Protocol decision D1.
- Stage-II shortlist rule of the 2026-09-11 re-run: baseline + 11 Pareto-front + top 14 on e_v(TS) + top 8 on e_v(DS) from the 89 II-a survivors.
- `shunt_int_g_w` rule: g_w,sh = T_p |g_H|_p90 / (2 (q/2 + δ) T_TS); T_p = 30 min → 0.030, 60 min → 0.059 at |g_H|_p90 = 0.133 (reference 2026-09-11, measured on the old placement).

## Follow-up: author decisions and instrumentation for D1 / D2b

Author decisions (entered in protocol §9): D1 per-bus RMS as the text reads; D2 T_p = 60 min; D2b re-measure; D3 no extra DS limit; D4 1.5-decade spans for `g_w_der`, `g_w_pcc`, `g_w_gen`; D6 all necessary re-runs; D8 seed 20260910; D9 shunt switching reported only; D10 name `ch9_selected_10_2026`. D7 open (author asked whether STS 0.015 pu is better).

| File | Change |
|---|---|
| `experiments/helpers/records.py` | New fields `dso_group_v_rms_dev_pu` (per-group spatial RMS of V − V^set over the DSO controllers' voltage buses, true values) and `zone_tso_shunt_g_h` (per-bank integrator pressure g_H on TS iterations). Defaulted, old pickles load. |
| `experiments/runners/_multi_tso_helpers.py` | `_record_dso_group_and_transformer_data`: accumulates Σ(V − V^set)² and bus count per group against `DSOControllerConfig.v_setpoints_pu`; writes `dso_group_v_rms_dev_pu`; no entry when a controller has no references; `ValueError` on a reference/bus length mismatch. |
| `experiments/runners/multi_tso_dso.py` | Local-DSO branch writes the same quantity (same buses `hv.bus_indices`, reference `v_setpoint_pu`, as the OFO DSOs use). Integrator block records `rec.zone_tso_shunt_g_h[z] = grad_g_list` before `integ.update`. |
| `experiments/helpers/comparison_metrics.py` | New `has_ds_rms_dev`, `voltage_rms_err_per_group_ds`, `voltage_rms_err_all_ds` (quadratic mean over groups, equal weights — the construction of the TS `voltage_rms_err_all`). No fallback inside, so a per-bus figure is never silently replaced by a mean. |
| `tuning/metrics.py` | `_v_quality`: `v_rms_ds` = per-bus figure when the log carries it, else the previous `|mean − v_set|` (old pickles keep their value). Docstring updated. |
| `experiments/ch_9_parameter_selection/ch_9_3_stage2_screen.py` | `--v-metric {areamean,bus}` (default `areamean` = every earlier campaign); every window row stores `e_v_*_areamean` and `e_v_*_bus`, and `e_v_ts` / `e_v_ds` carry the ranked one. `--span NAME=DECADES` overrides box spans (D4); resolved spans and `v_metric` written to `run_meta.json`. |
| `tests/test_ds_voltage_rms_dev.py` | New, 7 tests: recorder per-bus RMS (mean-on-setpoint profile gives 0 for the mean, 0.0245 for the RMS), per-bus references, no entry without references, length mismatch raises, quadratic mean over groups, `_v_quality` per-bus when recorded and unchanged for old logs. 7 passed. |

Smoke (scratch `smoke_d1_d2b.py`, `ch9_selected_pcc27`, 7 min): `dso_group_v_rms_dev_pu` in 21/21 records (e.g. DSO_4 0.0136 vs |mean − 1.03| 0.0119); `zone_tso_shunt_g_h` on TS iterations (t = 20, 180, 360 s; zone 2 |g_H| 0.02–0.11, zone 3 0.69–0.81 in the start-up transient); one 15-min Stage-II unit with `v_metric = bus`: e_v(TS) per-bus 0.00523 vs area-mean 0.00188, e_v(DS) 0.00810 vs 0.00677 — the two definitions differ materially, so D1 can change the selection.

D7 evidence (scratch `engage_check.py`, corrected commit rule on the frozen operating points): see protocol §9.3 — all runs so far engaged STS taps at ~1.7 % and TS taps at ~0.8 %; the stated 2 % / 1 % with the corrected rule would make both engage later.

## Execution under delegation (author away, 2026-10-16 ~16:30: "run all the reruns from the tuning campaign you need to run")

| File | Change |
|---|---|
| `tuning/stage1_screen.py` | `--d-engage-ts`, `--d-engage-sts` (defaults 0.010 / 0.020 unchanged) passed to `design_stage1`, so protocol input D7 is on the command line. |
| `experiments/ch_9_parameter_selection/ch_9_3_stage2_screen.py` | (above) plus `--span NAME=DECADES`. |
| `experiments/ch_9_parameter_selection/ch_9_3_shortlist.py` | New: II-a → II-b shortlist by the 2026-09-11 rule (baseline + Pareto on (e_v_ts, e_v_ds) + top 14 TS + top 8 DS among candidates passing `check_limits` without wear), writes a `--candidates-json` list plus `.meta.json` with each member's role. |
| `configs/paramsets/stage1_start_10_2026.json` | New, written by Stage I `--write-paramset`. |

- Protocol frozen for execution; D7 (TS 0.0075 / STS 0.015) and the D2b procedure set by Claude under the delegation and marked for author confirmation. Code state snapshot instead of a commit (no commit go-ahead): `results/THESIS_ch8_expB_10_2026/code_state/` (HEAD 781a819 + byte-exact patch, verified).
- Results so far in protocol §11. **D2b gave `shunt_int_g_w` = 0.570** (pooled p90 1.28, set by W4/W5; quiet windows ≈ 0.06–0.10) — about 10× the value in use; followed as frozen, flagged.
- II-a running (94 candidates × 6 × 3 h, est. 4.4 h).

## Overnight results (2026-10-17; full record in protocol §11)

- II-a 564/564 ok (3.7 h), 75/94 feasible; shortlist 23 (`ch_9_3_shortlist.py`); II-b 138/138 ok (3.2 h).
- Selection: 4/23 feasible (15 rejected for hunting); rule ranks the **Stage-I point** first. Experiment C: **all four feasible candidates FAIL** (p95 32–35 > 27, median residual 2.3–8.9 Mvar). **Protocol outcome: no admissible candidate; no `ch9_selected_10_2026` written; no rule relaxed.**
- Orientation (declared §4): `ch9_selected` (399e30d3b60b) and `ch9_selected_pcc27` are feasible on this configuration and dominate the Stage-I point on both voltage criteria — the 512-draw search in the widened 8-D box did not sample that region. The shunt weight 0.570 is not what makes candidates hunt.
- Diagnostic (outside the protocol, for the author's decision): archived 28-candidate shortlist re-simulated (162/162 ok, 4.7 h); union selection ranks pcc27 (hand-made, entered via orientation) first and 399e30d3b60b (search-generated) second; restricted to search-generated candidates the rule picks **399e30d3b60b**. Diagnostic Experiment C on both running.

## Run 2 and result (2026-10-19/20; protocol §10 Amendments 1–3, §11)

- Amendment 1 (author): D7 confirmed; D2b pooled over ranked windows → `shunt_int_g_w` 0.1605; box back to the default spans (the text's 0.5–1.5 decades). Amendment 2 (author): all 119 convergence-passing draws + starting weights at 12 h, no pre-screen/shortlist (as the Ch 8 text states). Amendment 3 (clarification, fixed before the deciding result was known): failed settling → repeat the selection with the settling limit, band re-anchored.
- New files: `configs/paramsets/stage1_start_10_2026_r2.json`, `configs/paramsets/ch9_selected_10_2026.json`.
- 45 / 119 feasible; ranking winner f879539ceec5 fails settling (p95 28); repeated selection → **df17c0c32904** (settling p95 23.05, residual 0.38 Mvar). **`ch9_selected_10_2026` ≈ `ch9_selected` within ±5.4 % per weight** (ratio 81): the procedure re-finds the September optimum on the final configuration.
- Compute: run 2 ≈ 4.0 h (II-a) + 3.3 h + 11.7 h (12 h bank) + 4 h (C).

## Switch to the result (2026-10-20, author decision)

| File | Change |
|---|---|
| `experiments/ch_9_parameter_selection/_ch9_selected_design.py` | `SELECTED_PARAMSET` "ch9_selected" → **"ch9_selected_10_2026"** (comment with protocol reference). Used by the ch_9_* scripts, `build_selected_config`, `make_config_ch9_frozen`. |
| `experiments/run_multi_system_ofo.py` | `main()` uses `SELECTED_PARAMSET` again (was the literal "ch9_selected_pcc27_gqceil_comp_q_off"); comment lists the hand-set variants of 2026-10-14..19, which remain as parameter sets. |
| `configs/paramsets/ch9_selected_10_2026.json` | `provenance.weight_fingerprint` recomputed over the nine weights only (df17c0c32904, as for `ch9_selected`); it had included `shunt_int_g_w`. |

Checks: script parses; main config ratio 81.1, shunt 0.1605; `build_selected_config()` → candidate_key df17c0c32904, DSO_3 ×2; `make_config_ch9_frozen()` ok; `tests/tuning/test_parameters.py` 17 passed. Not changed: `experiments/helpers/rms_cosim_config.py::DEFAULT_PARAMSET` ("ch9_selected") and `INITIAL_PARAMSET` ("stage1_start").

## Open

- Author: `rms_cosim_config.DEFAULT_PARAMSET` and `INITIAL_PARAMSET` (→ `stage1_start_10_2026_r2`?); commit; downstream re-runs (Ch 8 C/D1/D2 text, Ch 9–11); thesis text for Ch 8 (one sentence on the settling repetition; numbers from §11); STS-contribution preference → Ch 10/11 sensitivity (ratio sweep; runner-up 99ea756282bf).
