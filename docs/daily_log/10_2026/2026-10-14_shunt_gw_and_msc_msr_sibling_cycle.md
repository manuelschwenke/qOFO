# Shunt integrator g_w 0.03 -> 0.06, and the MSC/MSR sibling limit cycle

- Timestamp: 2026-10-14, 16:30–17:15 (CEST).
- Reason: Author observed too-frequent MSC/MSR dispatch with `ch9_selected` and asked for a reduction. Option chosen by the author: raise `shunt_int_g_w` only (no reversal-specific dwell). Author's hypothesis: reversals come from the dense contingency pattern of the short scenario.

## What changed

| File | Change |
|---|---|
| `configs/paramsets/ch9_selected_shunt_gw006.json` | New parameter set. `ch9_selected` (399e30d3b60b) with `shunt_int_g_w` 0.03 → 0.06; every other field identical. Commit persistence at the p90 pressure `|g_H| = 0.133`: `T = 2 g_w (q/2 + δ) / |g_H| · T_TS` = 30.5 → 60.9 min. Marked `closed_loop_validated: false`. `ch9_selected` itself unchanged. |

No code changed. Scratch scripts (session scratchpad, not in the repo): `shunt_gw_windows.py` (reuses `ch_9_3_stage2_screen._run_one_window` unchanged), `shunt_gw_analyse.py`, `trafo_activity.py`, `q_movement_breakdown.py`, `sibling_stats.py`.

## Runs

Results: `results/THESIS_shunt_gw_compare_20261014/` (`summary.json`, `windows/*.json`, `make_config/6h/records_*.pkl` + figures). Current code (voltage rescaling on, `int_cooldown` 5), dwell 30 min, δ = 10 Mvar, 4 tertiaries × (MSC + MSR), 2 × 25 Mvar each.

1. **W4, W5 evented, 12 h, DSO_3 ×2** (Stage-II evaluator; W4 = 4-step load ramp at 324–384 min, W5 = gen trip 288 min / restore 348 min).
2. **`make_config` scenario, 6 h, symmetric** (W1 start, 6 contingencies at 30/120/180/210/300/330 min; same as the author's run `run_multi_system_ofo/0006`).

| run | g_w | commits | rev ≤ 2 h | commits ≤ 30 min after event | e_v_ts | e_v_ds | e_q | v_worst_ts |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| W4 | 0.03 | 65 | 52 | 11 | 0.00221 | 0.00706 | 319 865 | 0.0195 |
| W4 | 0.06 | **41** | **28** | 4 | 0.00209 | 0.00701 | 200 521 | 0.0190 |
| W5 | 0.03 | 112 | 82 | 22 | 0.00655 | 0.00946 | 209 786 | 0.0424 |
| W5 | 0.06 | **57** | **33** | 12 | 0.00593 | 0.00820 | 163 127 | 0.0454 |
| 6 h | 0.03 | 14 | 1 | 12 | 5.0/7.8/4.6 mpu per zone | — | — | — |
| 6 h | 0.06 | **11** | 1 | 9 | 5.0/8.0/4.7 mpu per zone | — | — | — |

No corridor violations in any run. Baseline W4/W5 reproduce the Stage-II provenance (63/53, 96/66) closely, so today's code changes are not the cause.

- 0.06 reduces commits by 37 % (W4), 49 % (W5), 21 % (6 h) and **improves** e_v_ts, e_v_ds and e_q in W4 and W5; costs: W5 v_worst_ts +3 mpu; 6 h: TS zone 2/3 RMS error +1.5–3 %, DSO_2 max voltage 1.053 → 1.074 pu.
- Author's hypothesis holds for the 6 h scenario (12/14 commits within 30 min after an event, 1 reversal) but **not** for W4/W5: there 41/52 (W4) and 70/82 (W5) reversals are not near an event, and the median reversal gap equals the 30 min dwell.

## Finding: MSC and MSR at one tertiary cycle as a pair

Banks are built as (MSC, MSR) pairs at the same tertiary bus (`hv_networks.py` §4); each has its own integrator (`ShuntBank`). Their pressures are exact negatives (`sign_b`, same bus), and dwell, hysteresis and clamp are per bank. Measured (recorded state: MSC = −level, MSR = +level):

| run | share of commits made in the same instant as the sibling | sibling pairs moving 50 Mvar together | min with capacitor AND reactor energised | reversals executed by the sibling |
|---|--:|--:|--:|--:|
| W4 0.03 | 62 % | 20 | 309 | 24 |
| W4 0.06 | 49 % | 10 | 423 | 12 |
| W5 0.03 | 96 % | 54 | 0 | 40 |
| W5 0.06 | 91 % | 26 | 153 | 16 |
| 6 h 0.03 | 43 % | 3 | 51 | 0 |

Mechanism (established from code + traces, W4 tertiary `z2[2]/z2[3]`): after the MSC engages, the overshoot reverses the pressure; the MSC is in dwell, but the MSR is not, so the reversal is executed by the sibling (per-bank dwell circumvented, per-bank reversal metric blind to it), leaving both energised. Later the two integrators sit at symmetric states and commit in the same instant — one releases while the other engages — i.e. a 50 Mvar step from a device pair designed as 25 Mvar steps, which overshoots and cycles at the dwell rate. `shunt_int_g_w` slows this but cannot remove it; the two hottest banks in W4 drop only 16 → 14 and 17 → 14 commits.

Not changed here (architectural, for discussion): one integrator per tertiary over a signed net lattice ℓ ∈ {−N_C … +N_R}, release-before-engage, one dwell per tertiary.

## Side analysis: coupling-transformer activity (6 h baseline)

- T0/T3/T6/T9 host the tertiary banks (`coupling_lv_bus_indices[0]`). 79–94 % of their setpoint movement occurs at shunt-commit instants (feedforward ΔQ_itf); between commits all three transformers of a DSO move 4–12 Mvar (T3: 41, DSO_2 ramp). Transformers are electrically identical.
- Total |ΔQ| over 6 h: TS DER 1868, TS gen Q 1932, DSO DER 1450, shunts 350 (14 × 25), TSO PCC setpoints 216 Mvar. The TSO moves the PCC setpoints little because `ch9_selected` prices them `g_w_pcc / g_w_der` = 0.4936 / 0.00609 = 81× above TS DER (Stage-I point: 1×). The DSOs track what is asked (T1/T2-type RMSE 0.5–0.9 Mvar).
- DSO capability band reported for T0-type ≈ 74–87 Mvar vs 126–144 Mvar for the others (open-loop ∂Q_T/∂Q_DER); cause not verified.

## Follow-up 17:30: script switched, structural fix drafted

| File | Change |
|---|---|
| `experiments/run_multi_system_ofo.py` | `main()` applies `"ch9_selected_shunt_gw006"` (literal, with comment) instead of `SELECTED_PARAMSET`. Author decision. `SELECTED_PARAMSET` stays `ch9_selected`: it is the Ch-9 controller identity used by `ch_9_1_*`, `_ch9_selected_design` and `make_config_ch9_frozen`, none of which change. |
| `docs/architecture/2026-10-14_tertiary_shunt_dispatcher_DRAFT.md` | Draft of the per-tertiary MSC/MSR dispatcher (one integrator over a signed net level, release before engage, shared dwell/budget). For author review; not implemented. |

## Open

- Review the per-tertiary dispatcher draft (decisions Q1–Q5 there).
- `ch9_selected_shunt_gw006` not passed through the Ch-9 selection rule; W1–W3, W6 not run.
