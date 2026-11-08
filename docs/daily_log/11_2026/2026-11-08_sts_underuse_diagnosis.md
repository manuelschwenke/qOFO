# 2026-11-08 — Why the v4 weights leave the STS nearly idle (diagnosis, no code change)

**Trigger.** The author: with `trial_061d_v4_run0008` the STS reactive capability is used far less
than the TS IBG (≈ factor 100). Runs compared:
`results/run_multi_system_ofo/0030_2026-11-08_110721` (g_w_pcc = 0.10) and
`…/0031_2026-11-08_111611` (g_w_pcc = 0.87; the tuned value is 0.87296).
Both: make_config period 2016-01-05 06:00, 10 h, its events; DSO_3 ×2; code e5f6e4d.

## 1. The weights are the procedure's output
`trial_061d_v4_run0008.fields` = `ch9_selected_uc_2026_11_v4.fields` = run 0008
`selection_complete_sts+bank.json` ranked[0] (fingerprint 061dadc05755), plus
`sts_coupler_hold_after_bank = 1`. Its description text ("W5 pending") is stale.
Run 0031 used g_w_pcc = 0.87 (config.pkl), i.e. −0.34 % from the tuned value.

## 2. Mechanism (established model facts)
- TS step: min wᵀG_w w + ∇fᵀw (`build_miqp_problem`), g_u = 0, g_res = 0 → unconstrained
  Δu_i = −(g_v/g_w,i) Σ_j H_V[j,i] e_j. No usage price: allocation is set by G_w alone.
- Column sensitivities |Σ_j H_V[j,i]| (run-0008 operating points W1–W6): PCC/IBG = 0.93
  (zone 2), 1.30 (zone 3). Per Mvar the two act equally on TS voltages.
- So per Mvar the STS is priced g_w_pcc/g_w_der = 61.4 times the TS IBG.
- Analytic share of the first step (uniform deviation): σ_STS = 0.12 (zone 2), 0.03 (zone 3);
  participation-level ratio r = (ΔQ_STS/Q_cap,STS)/(ΔQ_IBG/Q_cap,IBG) = 0.04 / 0.08.
  Measured in 0031, zone 2: std(Q_TS-IBG)/std(Q_STS) = 67.3/9.4 = 7.2 vs predicted (1−σ)/σ = 7.3.

## 3. Why the selection landed there
- Stage I had g_w_pcc = g_w_der = 0.0897. The search moved g_w_pcc +0.99 decade (box top
  0.897) and g_w_der −0.80 decade.
- All 27 completed feasible candidates lie inside the 2 % e_v(TS) band, so the **e_v(DS)
  tie-break decided alone**. Spearman over the feasible set: log g_w_pcc vs e_v(DS) −0.76,
  vs e_v(TS) +0.51. The tie-break therefore selects the least STS use available.
- Feasibility (378 II-a/II-b candidates): g_w_pcc < 0.1 → 55/55 early-aborted;
  g_w_der ≥ 0.03 → 83–100 % aborted. Densest feasible cell: g_w_pcc 0.2–0.4, g_w_der
  0.005–0.03 (16/53). λ screen removes g_w_pcc ≲ 0.035.
- No criterion rewards STS **reactive** use; "STS engagement" in v3/v4 counts coupler moves.

## 4. Effect in the runs (one 10-h period, one seed)
| | 0030 (0.10) | 0031 (0.87) |
|---|---|---|
| e_v(TS) bus | 0.00459 | 0.00486 |
| e_v(DS) bus | 0.00673 | 0.00534 |
| e_q (ITAE) | 283 k | 51 k |
| zone 2 std Q: TS-IBG / STS [Mvar] | 44.9 / 51.3 | 67.3 / 9.4 |
| zone 2 min ↓ reserve TS-IBG [Mvar] | 79 | 26 |
| STS coupler moves / hunting | 13 / 0 | 7 / 0 |

In 0031 the zone-2 TS IBG falls to 26 Mvar absorbing reserve while ≥ 506 Mvar STS reserve is idle.

## 5. Re-selection potential inside run 0008 (no new simulation)
Feasible, completed, inside the e_v(TS) band: 1c2ff0dbe86a (g_w_pcc 0.199, g_w_der 0.0382,
r₂ 0.43, σ₂ 0.61, e_v_ts 0.00796, e_v_ds 0.00614) and c8c2f9f2cf70 (0.150 / 0.0274, r₂ 0.41).
Both sit at the edge of the feasible region (thin margin). Equal level r₂ = 1 needs
g_w_pcc ≈ 0.032 at g_w_der 0.0142, inside the region where every candidate hunted.

## Open
- Cause of hunting at low g_w_pcc (hypothesis: T_TS = 180 s vs STS settling p95 ≈ 23 × 20 s).
- Any network change for scarcity invalidates operating points (key netf8c419bf), λ, θ, r and
  all run-0008 results: decide the network first, then tune once.

Script/output: `2026-11-08_sts_participation_index.py` / `.out`. Author decision pending.
