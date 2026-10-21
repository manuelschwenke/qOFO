# g_w_pcc / g_w_der ratio sweep: how much the TSO asks of the STS

- Timestamp: 2026-10-15, 16:20–17:45 (CEST).
- Reason: Author observation in run `run_multi_system_ofo/0008`: zone-2 DSO DER do little while the TS DER makes steep moves. Analysis (same day) showed the cause is the price ratio `g_w_pcc / g_w_der = 0.4936 / 0.00609 = 81` of `ch9_selected` (Stage-I start point: 1). The Stage-II selection rule has no criterion on resource allocation, so the ratio is a by-product. Author asked for a sweep.
- No code changed. Scratch scripts (session scratchpad): `pcc_ratio_screen.py`, `pcc_ratio_screen2.py`, `pcc_ratio_sweep.py`, `pcc_ratio_summary.py`, `mc6h_detail.py`, `dso_der_util.py`, `zone_q_effort.py`.

## Run 0008 diagnosis (6 h, `ch9_selected_shunt_gw006`, new placement)

- Zone 2 per TSO iteration: TS DER total |ΔQ| 715 Mvar, largest step 132 Mvar (at the contingency instants 330/120/180/30 min); TSO PCC decision for the nine DSO_1–3 setpoints 206 Mvar in total.
- 250 Mvar load connect (120 min), 30 min later: TS DER +134, TSO PCC decision −54, shunt feed-forward −106, DSO DER −18 Mvar.

## Analytic screen (λ_max(B), DSO_3 ×2, W1–W6 operating points, `results/stage1/operating_points`)

| candidate | g_w_der | g_w_pcc | ratio | worst λ_max(B) |
|---|--:|--:|--:|--:|
| base | 0.00609 | 0.4936 | 81.1 | 1.3104 (= stored) |
| PCC only | 0.00609 | 0.1643 | 27 | 1.357 |
| PCC only | 0.00609 | 0.0846 | 13.9 | 1.546 (select ceiling, bisection) |
| PCC only | 0.00609 | 0.0548 / 0.0183 / 0.0061 | 9 / 3 / 1 | 1.84 / 3.49 / 8.48 (TS zone 2 @ W5) — rejected |
| balanced k = √(81.1/r) | 0.00609·k | 0.4936/k | 27 / 9 / 3 / 1 | 1.2751 each (binding row: DSO_4 @ W6, STS layer) |

Lowering only the PCC price adds TS-layer gain; shifting the ratio at roughly constant gain is unconstrained down to 1:1.

## Closed-loop sweep

`results/THESIS_pcc_ratio_sweep_20261015/` (`runs/*.json` + `*.pkl`, `summary.json`). Five candidates on `ch9_selected_shunt_gw006`: base, pcc27, pcc14, bal9, bal1. W1–W6 evented 12 h on DSO_3 ×2 (Stage-II evaluator settings) plus the `make_config` 6 h scenario; 35 units, all ok, 24–40 min each in parallel.

Ch-9 metrics, ranked windows (W1–W4, W6; W5 zero-capability excluded), `tuning.stage2_search.aggregate`:

| cand | ratio | e_v(TS) | e_v(DS) | e_q | viol | worst dev TS / DS | DSO taps worst /d |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 81 | 0.00135 | 0.00477 | 112 780 | 0 | 0.0191 / 0.0285 | 4 |
| pcc27 | 27 | 0.00129 (−4.4 %) | 0.00526 (+10 %) | 134 359 (+19 %) | 0 | 0.0186 / 0.0350 | 6 |
| pcc14 | 13.9 | 0.00126 (−6.7 %) | 0.00604 (+27 %) | 195 826 (+74 %) | 0 | 0.0183 / 0.0536 | 4 |
| bal9 | 9 | 0.00141 (+4.4 %) | 0.00501 (+5 %) | 137 200 (+22 %) | 0 | 0.0186 / 0.0306 | 6 |
| bal1 | 1 | 0.00146 (+8 %) | 0.00550 (+15 %) | 318 465 (+182 %) | 0 | 0.0182 / 0.0529 | 6 |

Allocation, ranked windows (sums over the five windows):

| cand | TS DER total \|ΔQ\| | TS DER largest step | TSO PCC decision \|ΔQ\| | interface RMSE worst / mean | DSO DER \|ΔQ\| | STS-IBG utilisation mean / p95 | shunt commits | DS V max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 5446 | 94 | 813 | 2.8 / 1.1 | 9276 | 0.180 / 0.53 | 40 | 1.06 |
| pcc27 | 5163 | 92 | 1511 | 4.0 / 1.4 | 9518 | 0.185 / 0.56 | 26 | 1.07 |
| pcc14 | 5312 | 90 | 2335 | 5.2 / 2.2 | 10043 | 0.191 / 0.56 | 19 | 1.08 |
| bal9 | 3255 (−40 %) | 53 (−44 %) | 1895 | 4.0 / 1.4 | 9774 | 0.178 / 0.55 | 30 | 1.07 |
| bal1 | 1576 (−71 %) | 71 | 3779 | 8.9 / 3.1 | 11139 | 0.193 / 0.56 | 17 | 1.09 |

6 h scenario: same ordering; bal9 TS v_rms z1/z2/z3 5.18/9.11/4.62 vs base 5.01/8.99/4.60 mpu, zone-2 TS DER largest step 132 → 69 Mvar; bal1 reaches DS V max 1.095 pu.

Findings:
- Lowering only `g_w_pcc` doubles/triples the PCC movement and improves e_v(TS) by 4–7 %, but leaves the TS DER as steep as before; it costs e_v(DS) +10/+27 % and e_q +19/+74 %.
- Only the balanced shift moves effort off the TS DER (bal9: total −40 %, largest step −44 %) at e_v(TS) +4.4 %, e_v(DS) +5 %, e_q +22 %.
- STS-IBG time-mean utilisation is unchanged in all candidates (0.18–0.19): the extra demand appears as interface movement during events, not as a sustained shift of the DSO operating point.
- Interface tracking error rises with every lower ratio, consistent with DSO settling (p95 ≈ 7 min) being slower than the TS period (3 min).
- The "share of DSO capability used" metric was dropped: the DSO reports its band as a delta around the current flow (`dso_controller.py` l. 573), so the measure reflects tracking lag, not capability use.

Baseline differs from the stored `ch9_selected` aggregate (0.001456 / 0.005177 / 120 277): new MSC placement, `shunt_int_g_w` 0.06, `int_cooldown` 5, shunt voltage rescaling.

## Follow-up 17:10–18:45: balanced ratio 27 (`bal27`, author request)

`g_w_der` 0.01055 (×1.73), `g_w_pcc` 0.2848 (÷1.73); λ_max(B) 1.2751 (STS-limited). First attempt: 6 h unit ok, the six window units lost when a pool worker died with an ntdll thread-pool fault (0xc000070a, Windows Error Reporting 17:52) while the pool still held 35 idle workers for cached units; driver changed to start workers only for missing units, re-run clean.

| cand | ratio | e_v(TS) | e_v(DS) | e_q | DSO taps worst /d | TS DER \|ΔQ\| | largest TS-DER step | PCC \|ΔQ\| | itf RMSE worst / mean | shunt commits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 81 | 0.00135 | 0.00477 | 112 780 | 4 | 5446 | 94 | 813 | 2.8 / 1.1 | 40 |
| bal27 | 27 | 0.00139 (+3.0 %) | 0.00502 (+5.2 %) | 119 648 (+6.1 %) | 4 | 4162 (−24 %) | 73 (−23 %) | 1240 (×1.5) | 3.2 / 1.2 | 36 |
| bal9 | 9 | 0.00141 (+4.4 %) | 0.00501 (+5.0 %) | 137 200 (+22 %) | 6 | 3255 (−40 %) | 53 (−44 %) | 1895 (×2.3) | 4.0 / 1.4 | 30 |

STS-IBG utilisation bal27: mean 0.177 (base 0.180). 6 h scenario bal27: TS v_rms 5.10/9.33/4.56 mpu, zone-2 TS DER largest step 102 Mvar (base 132, bal9 69).

Reading: the TS and DS voltage costs of bal27 and bal9 are almost equal; what separates them is interface tracking (e_q +6 % vs +22 %, DSO taps 4 vs 6 /d) against TS-DER smoothing and interface use (−24 % vs −40 %, ×1.5 vs ×2.3).

## Follow-up 2026-10-16: goal restated, recruited flow

Author: the goal is more STS contribution; reducing TS-DER activity is not required. Contribution measured with the Ch 7 recruited flow (`eq:setup:metrics:recruit`), the base weights taking the place of the no-access variant: per STS, T[Σ_j |Q_STS,j^cand − Q_STS,j^base|] over its three interfaces, median over STS × scenarios (scratch `recruited_vs_base.py`). Per-unit DSO DER Q is not recorded (`dso_q_der` empty), so the Ch 7 STS-IBG utilisation is available only at group level (|ΣQ| / rating, a lower bound).

| cand | recruited flow vs base, windows / 6 h [Mvar] | DSO DER output difference, windows / 6 h | e_v(TS) | e_v(DS) | e_q |
|---|--:|--:|--:|--:|--:|
| pcc27 | 15.7 / 16.8 | 9.0 / 15.6 | −4.4 % | +10 % | +19 % |
| pcc14 | 19.4 / 27.0 | 14.6 / 30.9 | −6.7 % | +27 % | +74 % |
| bal27 | 7.9 / 9.3 | 4.1 / 6.9 | +3.0 % | +5.2 % | +6 % |
| bal9 | 15.8 / 22.2 | 10.2 / 18.6 | +4.4 % | +5.0 % | +22 % |
| bal1 | 27.8 / 46.9 | 27.6 / 54.2 | +8 % | +15 % | +182 % |

Reading: for the restated goal the PCC-only family is the natural one; pcc27 recruits as much as bal9 with better instead of worse TS tracking, at a higher DS voltage cost (+10 % vs +5 %); shunt commits 40 → 26. With the tracking objective the total reactive effort is set by the voltage errors and the ratio only decides the split, so a sustained STS share needs an objective term, not a price.

## Adopted 2026-10-16 (author decision): pcc27

| File | Change |
|---|---|
| `configs/paramsets/ch9_selected_pcc27.json` | New parameter set: `ch9_selected_shunt_gw006` with `g_w_pcc` 0.4936 → 0.16431 (= 27 × `g_w_der`); all other fields unchanged. Provenance carries the sweep, λ_max(B) 1.357 and the ranked-window aggregate. Verified field-for-field identical to the weights simulated as `pcc27`. |
| `experiments/run_multi_system_ofo.py` | `main()` applies `"ch9_selected_pcc27"` (was `"ch9_selected_shunt_gw006"`); comment updated. `SELECTED_PARAMSET` stays `ch9_selected`. |

## Follow-up 2026-10-16: raise the STS interface weight g_q as well? (analysis, no change)

- Interface error e = Q_act − Q_set per DSO, ranked windows (scratch `itf_error_split.py`): base RMS 1.35 Mvar (1.41 in the first three DSO iterations after a TSO update, 1.38 at the next TSO step); pcc27 1.72 (1.80 / 1.72). The error barely decays within a TSO period. DSO DER at their rating in only 1.1–1.5 % of the top-decile error samples, so capability is not the cause.
- g_q headroom on `ch9_selected_pcc27` (scratch `gq_screen.py`): STS layer binds at DSO_4 @ W6; λ_max(B) 1.275 (×1.0), 1.388 (×1.1), 1.502 (×1.2), 1.593 (×1.28), 1.731 (×1.4). Select ceiling 1.546 is reached at about ×1.22.
- Earlier evidence (2026-09-29 D settling): g_q ×1.28 shortened p95 interface settling 23.05 → 20.05 DSO iterations (−13 %); per-area ceiling 16.05.
- Reading: a global g_q raise is capped at ~+20 % by the DSO_4 row, buys roughly 10 % on interface settling, spends most of the STS contraction margin and shifts DSO priority from voltage to Q. Not adopted; a per-area raise for DSO_1–3 (not binding) is the better test if tracking is to be tightened.

## Follow-up 2026-10-16: per-area g_q for DSO_1–3 on pcc27 (author request)

Screen (scratch `gq_perarea_screen.py`, method of `ch_8_D1_settling_screen.analytic`: STS rows re-screened per area; TS rows verified independent of g_q): λ_max(B) at ×1 DSO_1 1.008, DSO_2 1.088, DSO_3 0.967, DSO_4 1.275; per-area select-ceiling multipliers ×1.556 / ×1.448 / ×1.617 / ×1.238. At ×1.25: 1.249 / 1.343 / 1.201, i.e. below the binding TS row (1.357), so the overall certificate is unchanged.

Driver extended with config overrides (`CFG_OVERRIDES`, `dso_g_q_per_area`); smoke run confirmed `[dso_g_q] DSO_1: 0.497695 -> 0.7747` etc., DSO_4 untouched. 14 units, all ok.

| cand | g_q DSO_1/2/3 | worst λ | e_v(TS) | e_v(DS) | e_q | interface RMS (all / at next TS step) | recruited flow windows / 6 h |
|---|---|--:|--:|--:|--:|--:|--:|
| base | 0.498 | 1.310 | 0.00135 | 0.00477 | 112 780 | 1.35 / 1.38 | — |
| pcc27 | 0.498 | 1.357 | 0.00129 | 0.00526 | 134 359 | 1.72 / 1.72 | 15.7 / 16.8 |
| pcc27_gq125 | 0.622 ×3 | 1.357 | 0.00129 | 0.00539 (+2.5 %) | 122 119 (−9 %) | 1.65 / 1.65 | 15.7 / 19.0 |
| pcc27_gqceil | 0.775 / 0.721 / 0.805 | 1.546 | 0.00129 | 0.00540 (+2.7 %) | 112 964 (−16 %, = base) | 1.60 / 1.59 | 15.8 / 19.0 |

(% relative to pcc27.) No violations; DSO taps worst 6 /d unchanged; 6 h TS v_rms gq125 = gqceil 4.59/8.95/4.34 mpu (pcc27 5.04/9.06/4.38).

Reading: per-area g_q improves how well the DSOs deliver what is asked (e_q, interface RMS) without changing how much is asked (recruited flow in windows unchanged) or TS tracking; cost e_v(DS) +2.5–2.7 %. gq125 gets a little over half the e_q recovery with the contraction certificate unchanged; gqceil restores e_q to the base level but puts DSO_1–3 at the select ceiling.

## Adopted 2026-10-16 (author decision): pcc27_gqceil

| File | Change |
|---|---|
| `configs/paramsets/ch9_selected_pcc27_gqceil.json` | New parameter set: `ch9_selected_pcc27` + `dso_g_q_per_area` = {DSO_1 0.7747, DSO_2 0.7206, DSO_3 0.8046}; DSO_4 keeps global `g_q` 0.4977. Provenance: per-area ceilings, λ (overall 1.546 = select ceiling, TS 1.357, DSO_4 1.275), ranked-window aggregate against pcc27 and gw006. Verified identical to the simulated `pcc27_gqceil` weights and overrides. |
| `experiments/run_multi_system_ofo.py` | `main()` applies `"ch9_selected_pcc27_gqceil"` (was `"ch9_selected_pcc27"`); comment lists both previous values. |
| `configs/config.py` | `dso_g_q_per_area` docstring: "exists only to complete the voltage relief" replaced by two uses — (1) per-area interface-tracking speed (this change), (2) completing the relief; the γ = 0 remark now says the DER block still responds. No behaviour change. |

Validation: parameter set loads and resolves (`g_w_pcc/g_w_der` 27.0, per-area map present, relief off); `tests/test_dso_v_relief_pairing.py` 10 passed, 2 skipped; `tests/tuning/test_parameters.py` 17 passed. Collected together in one pytest call the two files aborted at import with `OSError: llvmlite.dll` (numba dependency); each passes on its own — environment issue, not investigated.

## Added 2026-10-16: ch9_selected_pcc27_gq125 (author request)

`configs/paramsets/ch9_selected_pcc27_gq125.json`: `ch9_selected_pcc27` + `dso_g_q_per_area` = 1.25 × 0.4977 = 0.62212 for DSO_1–3 (DSO_4 global). Provenance: per-area λ 1.249 / 1.343 / 1.201, overall 1.357 (TS row, unchanged), sweep aggregate. Verified identical to the simulated `pcc27_gq125`. Not applied to any script (`run_multi_system_ofo.py` stays on `ch9_selected_pcc27_gqceil`).

Observation for the harmonisation question: `ch9_selected` lies at the corner of the Stage-II box in the allocation direction — offsets from the Stage-I point `stage1_start_dso3x2` in decades of the ±span: g_w_der −0.94/1.0, g_w_pcc +0.97/1.0, g_w_gen −0.81/1.0, g_q +1.09/1.5; all others ≤ 59 %. The ratio 81 is therefore largely set by the box limits, not an interior optimum of the selection rule.

## Open

- Ch 9 / Ch 10 should state the PCC price as an explicit priority rule (author priority: STS contribution), since the Ch-9 selection rule alone would rank ratio 13.9 first.
- A sustained STS share (beyond event response) would need an objective term (cf. `tso_g_res_sg`), not a price ratio.
