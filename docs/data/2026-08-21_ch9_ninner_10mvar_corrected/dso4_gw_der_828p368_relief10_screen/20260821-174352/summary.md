# N_inner on the isolated subordinate loop (thesis eq. 9.2)

`N_inner` is the first subordinate iteration after an interface-Q setpoint step from which the per-iteration flow change `|q_pcc(k) - q_pcc(k-1)|` stays within the configured flatness tolerance. Right-censored at the end of the observation window; censoring is reported, never dropped.

## This is a check on the guess, not an independent measurement

`G_w` was calibrated with `N_inner = 9` **assumed**, and this measures `N_inner` with those weights in place. That is a fixed-point argument evaluated at one iteration: the guess is used, the weights follow, and this tests whether the guess survives its own consequence. Quote it that way. If the answer exceeds 9 the choice -- raise `T_TS`, or recalibrate the weights and repeat -- is an author decision, not a script output.

## Result

### Capability-limited steps are separated, not mixed in

A step is **capability-limited** when the subordinate layer's own reported headroom in the commanded direction, at the end of the window, is below the settling band -- it is saying it cannot move a further band-width that way. Such a step failing to settle measures the limit, not the loop. `N_inner` is read over the **free** steps.

| DSO | direction | steps | cap-limited | free | free N_inner med | free p95 | free censored | headroom med [Mvar] | step med [Mvar] |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_4 | down | 15 | 0 | 15 | 8.0 | 12.8 | 0.00 | 38.72 | 10.0 |
| DSO_4 | up | 15 | 0 | 15 | 9.0 | 16.0 | 0.00 | 35.75 | 10.0 |
| __pooled__ | both | 30 | 0 | 30 | 8.5 | 16.0 | 0.00 | 37.39 | 10.0 |

### All steps, including capability-limited ones

| DSO | direction | steps | N_inner med | p95 | max | censored | step med [Mvar] | band width med [Mvar] | residual med [Mvar] | taps | V viol |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DSO_4 | down | 15 | 8.0 | 12.8 | 17 | 0.00 | 10.0 | 107.9 | 0.08 | 0 | 0 |
| DSO_4 | up | 15 | 9.0 | 16.0 | 16 | 0.00 | 10.0 | 85.5 | 0.06 | 0 | 0 |
| __pooled__ | both | 30 | 8.5 | 16.0 | 17 | 0.00 | 10.0 | 96.7 | 0.07 | 0 | 0 |

## Windows excluded (no admissible traversal)

The DER reactive capability is structurally zero in these windows (VDE dead zone), so there is no capability band to traverse and no `N_inner` to measure. They are excluded, not failed: counting them would put a spurious censoring fraction into eq. (9.2).

- `d_quiet_summer`
- `d_ramp_up_winter`

## Provenance

- run: `2026-08-21T17:43:53.802731+02:00`
- script SHA-256: `28a273386899d6e85a9e1dd07447ea4f355b2fadf0dafe4f850d92f71fa9398a`
- commit: `5d8e0f31c2aa5b4256782f6709ce4516142a2bdf` on `main`  **(working tree dirty -- not reproducible from the commit alone)**
- parent-silent variant: **frozen_ofo** (`frozen_ofo` = supervisory OFO solves once at t=0 and holds; `local` = local Q(V) supervisory control, a DIFFERENT baseline controller and a cross-check only)
- weights: campaign `stage1` candidate `fe010aa3ead1`, rebuilt weights == archived weights
- settle 600.0 s, observe 900.0 s, tracking band 1.0 Mvar, flatness tolerance 0.1 Mvar, raw target 0.95 of the reported band, applied step capped at 10.0 Mvar
- `T_STS` = 20 s, bank `tier1_design_set` on `rural_700`
- command: `Z:\Python_Projekte\qOFO_GH\results\_codex_ninner_gwd_20260821\experiments\ch_9_parameter_selection\ch_9_1_ninner_isolated_sts.py --workers 16 --max-step-mvar 10 --case-design balanced_half --dso-v-authority-override 10 --dso-der-weight-override 828.3682036634343 --dsos DSO_4 --out Z:\Python_Projekte\qOFO_GH\docs\data\2026-08-21_ch9_ninner_10mvar_corrected --label dso4_gw_der_828p368_relief10_screen`